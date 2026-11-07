"""Model-scoped cross-recording prototypes, updated only from confirmed anchors."""
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
import numpy as np


def update_bank(path, model_scope, recording, tracks, groups, edges, mapping,
                similarity=.92, margin=.04):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    confirmed={i for e in edges if e['state']=='confirmed' for i in (e['i'],e['j'])}
    identities=[]
    for group in groups:
        samples=[p for i in group if i in confirmed and tracks[i].anchor for p in tracks[i].prototypes]
        if not samples:
            continue
        person=mapping[tracks[group[0]].key]
        identities.append((person,samples))
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript('''CREATE TABLE IF NOT EXISTS people(id INTEGER PRIMARY KEY, model TEXT);
            CREATE TABLE IF NOT EXISTS prototypes(person INTEGER, vector TEXT, support REAL);
            CREATE TABLE IF NOT EXISTS samples(model TEXT, source TEXT, PRIMARY KEY(model,source));
            CREATE TABLE IF NOT EXISTS assignments(model TEXT, recording TEXT, local_person INTEGER,
                person INTEGER, PRIMARY KEY(model,recording,local_person));''')
        db.execute('BEGIN IMMEDIATE')
        assigned=dict(db.execute('SELECT local_person,person FROM assignments WHERE model=? AND recording=?',(model_scope,recording)))
        # Frozen references exclude prototypes from this recording to avoid self reinforcement.
        references={}
        for person,vector,support in db.execute('SELECT p.person,p.vector,p.support FROM prototypes p JOIN people i ON i.id=p.person WHERE i.model=?',(model_scope,)):
            references.setdefault(person,[]).append(np.array(json.loads(vector)))
        claimed=set(assigned.values())
        output={}
        for local,samples in identities:
            person=assigned.get(local)
            if person is None:
                vectors=[np.array(s['e']) for s in samples]
                ranked=[]
                for candidate,prototypes in references.items():
                    if candidate in claimed:
                        continue
                    scores=sorted((float(v@p) for v in vectors for p in prototypes if v.shape==p.shape),reverse=True)[:3]
                    if scores:
                        ranked.append((float(np.mean(scores)),candidate))
                ranked.sort(reverse=True)
                if ranked and ranked[0][0]>=similarity and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=margin):
                    person=ranked[0][1]
                else:
                    person=db.execute('INSERT INTO people(model) VALUES (?)',(model_scope,)).lastrowid
                db.execute('INSERT INTO assignments VALUES (?,?,?,?)',(model_scope,recording,local,person))
                claimed.add(person)
            output[local]=person
            for sample in samples:
                key=hashlib.sha256((recording+':'+sample['source']).encode()).hexdigest()
                if db.execute('INSERT OR IGNORE INTO samples VALUES (?,?)',(model_scope,key)).rowcount==0:
                    continue
                vector=np.array(sample['e'],dtype=float)
                vector/=max(np.linalg.norm(vector),1e-12)
                q=max(float(sample['q']),1e-6)
                existing=[(rid,np.array(json.loads(v)),support) for rid,v,support in db.execute('SELECT rowid,vector,support FROM prototypes WHERE person=?',(person,))]
                compatible=[(float(vector@v),rid,v,support) for rid,v,support in existing if v.shape==vector.shape]
                best=max(compatible,key=lambda p:p[0]) if compatible else None
                if best and best[0]>=.95:
                    _,rid,old,support=best
                    updated=(support*old+q*vector)/(support+q)
                    updated/=max(np.linalg.norm(updated),1e-12)
                    db.execute('UPDATE prototypes SET vector=?,support=? WHERE rowid=?',(json.dumps(updated.tolist()),support+q,rid))
                elif len(existing)<12:
                    db.execute('INSERT INTO prototypes VALUES (?,?,?)',(person,json.dumps(vector.tolist()),q))
        return output
