"""Compara el análisis de escena de videos/prueba-actividades.mp4 con la actividad real de cada tramo."""
import json, sys, numpy as np
sys.path.insert(0, ".")
from analyze import ACT_LABELS as A
keys = [a['key'] for a in A]; es = {a['key']: a['es'] for a in A}
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'videos/.results/prueba-actividades.mp4.scene.json'))
gt = json.load(open('clips/prueba-actividades.json'))
print("cortes detectados:", r.get('cuts'))
ok_n = 0
for g in gt:
    P = []
    for s in r['samples']:
        if not (g['start'] + 0.3 <= s['t'] < g['end']):
            continue
        ppl = [o for o in s['objects'] if o['cls'] == 'person' and o['sims']]
        if ppl:  # persona más grande del fotograma
            o = max(ppl, key=lambda o: (o['box'][2] - o['box'][0]) * (o['box'][3] - o['box'][1]))
            v = np.array(o['sims']); p = np.exp((v - v.max()) * 60); P.append(p / p.sum())
    if not P:
        print(f"{g['start']:>4.0f}s {es[g['activity']]:<10} -> sin personas detectadas ❌"); continue
    mp = np.mean(P, 0); top = mp.argsort()[::-1][:3]; pred = keys[top[0]]; ok = pred == g['activity']; ok_n += ok
    print(f"{g['start']:>4.0f}s {es[g['activity']]:<10} -> {es[pred]:<12} {'✅' if ok else '❌'}  " +
          ", ".join(f"{es[keys[k]]} {mp[k]*100:.0f}%" for k in top) + f"   ({g['note']})")
print(f"aciertos: {ok_n}/{len(gt)}")
