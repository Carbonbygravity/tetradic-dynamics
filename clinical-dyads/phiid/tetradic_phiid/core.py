import hashlib
import json
from pathlib import Path
import numpy as np
from .vendor.phyid import calculate, utils

ATOMS = ('rtr', 'xtx', 'yty', 'sts')
MEANINGS = {'rtr': ('Red', 'Red'), 'xtx': ('Un1', 'Un1'),
            'yty': ('Un2', 'Un2'), 'sts': ('Syn', 'Syn')}
LABELS = {'rtr': 'Redundancy-to-redundancy', 'xtx': 'Leader-unique-to-Leader-unique',
          'yty': 'Follower-unique-to-Follower-unique', 'sts': 'Synergy-to-synergy'}
CELLS = [('Leader', 'Mirroring'), ('Leader', 'Matching'),
         ('Follower', 'Matching'), ('Follower', 'Mirroring')]

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True)

def verify_backend():
    root = Path(__file__).parent
    hashes = json.loads((root/'backend_hashes.json').read_text())
    for name, expected in hashes.items():
        require(digest(root/'vendor/phyid'/name) == expected, 'Backend source hash changed: '+name)
    for key, meaning in MEANINGS.items():
        require(utils.PhiID_atoms_abbr2full[key] == meaning, 'Atom meaning mismatch: '+key)
    return hashes

def validate_manifest(m):
    require(m['schema_version'] == 1, 'Unsupported manifest schema')
    require(bool(m['session_id']) and bool(m['assignment_evidence']), 'Session and assignment evidence required')
    ids = [p['id'] for p in m['participants']]
    require(len(ids) == 2 and len(set(ids)) == 2, 'Two distinct participant IDs required')
    for p in m['participants']:
        require(bool(p['recording_system']) and bool(p['identity']) and bool(p['file']), 'Participant identity, system and file required')
    require(len({p['recording_system'] for p in m['participants']}) == 2, 'Recording systems must differ')
    require(bool(m['blocks']) and bool(m['events']), 'Explicit blocks and events required')
    for block, row in m['blocks'].items():
        require(set([row['leader'], row['follower']]) == set(ids), 'Invalid actor assignment: '+block)
        require(row['condition'] in ('Mirroring', 'Matching'), 'Unknown condition')
    for label, row in m['events'].items():
        require(row['block'] in m['blocks'], 'Unknown block for event '+label)
        require(row['period'] in ('Leader', 'Follower'), 'Unknown response period')
    require(m['input_representation'] == 'raw_band_mean_power_float32', 'Expected unlogged frequency-mean power')
    require(bool(m['power_preparation']), 'Power preprocessing provenance required')
    require(m['epoch_anchor'] == 'response_period_onset', 'Epoch anchor must be explicit')
    require(m['pooling'] == 'all_eligible_paired_epochs', 'Unsupported pooling')
    require(m['offset_samples'] == 1 and m['min_trials'] == 10, 'This version preserves offset=1 and minimum=10')
    require(m['sampling_hz'] > 0, 'Positive sampling frequency required')

def load_inputs(manifest_path):
    path = Path(manifest_path)
    m = json.loads(path.read_text(encoding='utf-8'))
    validate_manifest(m)
    sources = [(path.parent/p['file']).resolve() for p in m['participants']]
    require(sources[0] != sources[1], 'Both participants cannot use the same input file')
    arrays = []
    for source in sources:
        with np.load(source, allow_pickle=False) as f:
            arrays.append({key: f[key] for key in f.files})
    a, b = arrays
    metadata = ['epoch_id', 'block_id', 'event_label', 'event_sample', 'roi_labels', 'band_labels', 'times_ms']
    for key in metadata:
        require(np.array_equal(a[key], b[key]), 'Paired alignment mismatch: '+key)
    n = len(a['epoch_id'])
    require(n > 0 and len(set(a['epoch_id'].tolist())) == n, 'Epoch IDs must be unique and nonempty')
    for key in ['block_id', 'event_label', 'event_sample']:
        require(a[key].shape == (n,), 'Epoch metadata shape mismatch: '+key)
    require(np.all(np.diff(a['event_sample']) >= 0), 'Events must be chronological')
    for f in arrays:
        require(f['power'].dtype == np.float32, 'Power must be float32 before log1p')
        require(f['power'].shape == (n, len(f['roi_labels']), len(f['band_labels']), len(f['times_ms'])), 'Power axes mismatch')
        require(np.isfinite(f['power']).all() and (f['power'] >= 0).all(), 'Nonfinite or negative raw power')
    for key in ['roi_labels', 'band_labels']:
        require(len(set(a[key].tolist())) == len(a[key]), 'Duplicate labels: '+key)
    require(np.array_equal(a['band_labels'], np.array(m['band_labels'])), 'Band definition mismatch')
    require(len(a['times_ms']) >= 3 and np.allclose(np.diff(a['times_ms']), 1000/m['sampling_hz']), 'Time grid mismatch')
    centres = m['centre_indices']
    require(bool(centres) and len(set(centres)) == len(centres), 'Unique centres required')
    require(all(isinstance(t, int) and 1 <= t < len(a['times_ms'])-1 for t in centres), 'Centre lacks past/future sample')
    mapping = actor_rows(m, a)
    counts = [sum((r['period'], r['condition']) == cell for r in mapping) for cell in CELLS]
    require(min(counts) >= 10, 'Fewer than 10 paired epochs in a response-period/condition group')
    return m, arrays, sources, mapping

def actor_rows(m, data):
    """No participant inference from event prefixes, suffixes or array position."""
    rows = []
    for epoch, block, event in zip(data['epoch_id'], data['block_id'], data['event_label']):
        require(event in m['events'], 'Unknown event: '+str(event))
        definition = m['events'][event]
        require(str(block) == definition['block'], 'Event/block disagreement: '+str(event))
        assignment = m['blocks'][str(block)]
        rows.append(dict(epoch_id=str(epoch), block=str(block), event=str(event),
                         period=definition['period'], condition=assignment['condition'],
                         X=assignment['leader'], Y=assignment['follower']))
    return rows

def observations(m, arrays, rows, cell, roi, other_roi, centre):
    powers = {p['id']: a['power'] for p,a in zip(m['participants'], arrays)}
    selected = [i for i,r in enumerate(rows) if (r['period'],r['condition']) == cell]
    # Return bands x [Xpast,Ypast,Xfuture,Yfuture] x paired epochs.
    return np.log1p(np.stack([np.stack([powers[rows[i]['X']][i,roi,:,centre-1],
                              powers[rows[i]['Y']][i,other_roi,:,centre-1],
                              powers[rows[i]['X']][i,roi,:,centre+1],
                              powers[rows[i]['Y']][i,other_roi,:,centre+1]], axis=1)
                     for i in selected], axis=2))

def estimate(v):
    require(v.ndim == 2 and v.shape[0] == 4 and v.shape[1] >= 10 and np.isfinite(v).all(), 'Invalid estimator observations')
    v = v[:,np.isfinite(v).all(axis=0)]
    sd = np.std(v, axis=1, ddof=1, keepdims=True)
    require((sd > 0).all(), 'Zero variance; no automatic regularization')
    v = v/sd
    h = calculate._get_entropy_four_vec(v, 'gaussian')
    i = calculate._get_coinfo_four_vec(h)
    res = dict(h_res=h, I_res=i, R_res=calculate._get_redundancy_four_vec('MMI', i))
    res['rtr'] = calculate._get_double_redundancy_four_vec('MMI', res)
    atoms = calculate._get_atoms_four_vec(res)
    result = np.array([np.mean(atoms[k]) for k in ATOMS])
    require(np.isfinite(result).all(), 'Nonfinite atom output')
    return result
