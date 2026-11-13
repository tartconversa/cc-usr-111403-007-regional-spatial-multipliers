"""现有业务摘要，不推断未观察到的业务关系。"""
def analyze(snapshot):
    cities=sorted(snapshot.rows('cities'),key=lambda r:r['id']);keys=[r['id'] for r in cities];positions={k:i for i,k in enumerate(keys)}
    w=[[0 for _ in keys] for _ in keys]
    for e in snapshot.rows('edges'):
        if e['from'] not in positions or e['to'] not in positions or e['from']==e['to'] or e['weight']<0:raise ValueError('空间边无效')
        w[positions[e['from']]][positions[e['to']]]+=e['weight']
    for i,row in enumerate(w):
        total=sum(row)
        if total:w[i]=[x/total for x in row]
    return {'cities':keys,'W':w,'y':[r['y'] for r in cities]}
