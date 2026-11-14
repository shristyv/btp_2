"""Step 16: interactive 2D FL demo. Playback of REAL saved validation log values;
vehicle movement and message animation are illustrative, not live training/detection.
"""
import argparse,csv,math
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--strategy',choices=['iid','mild_non_iid','strong_non_iid'],default='strong_non_iid')
    ap.add_argument('--labels',choices=['10','20'],default='10')
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    try:import pygame
    except ImportError:raise SystemExit('Install simulator dependency: uv pip install pygame')
    log=a.root.resolve()/f'results/phase2_supervised_v2/{a.strategy}/labels_{a.labels}/seed_{a.seed}/round_metrics.csv'
    if not log.exists():raise SystemExit(f'Missing real results: {log}\nRun Step 14 first.')
    with log.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    data={m:sorted((r for r in rows if r['method']==m),key=lambda r:int(r['round'])) for m in ['fedavg','fedprox','fedadam']}
    pygame.init();screen=pygame.display.set_mode((1100,700));clock=pygame.time.Clock()
    pygame.display.set_caption('BTP Part 2 — Four Vehicle Federated Learning Demo')
    font=pygame.font.SysFont('arial',19);large=pygame.font.SysFont('arial',26,bold=True)
    colors=[(80,175,235),(245,165,92),(143,210,150),(212,145,216)]
    chosen=0;roundindex=0;playing=False;tick=0
    def text(s,x,y,col=(235,241,248),big=False):screen.blit((large if big else font).render(s,True,col),(x,y))
    def rect(r,c,rad=8):pygame.draw.rect(screen,c,r,border_radius=rad)
    while True:
        dt=clock.tick(45)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:pygame.quit();return
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE:pygame.quit();return
                if event.key in (pygame.K_1,pygame.K_2,pygame.K_3):chosen=event.key-pygame.K_1;roundindex=0;tick=0
                if event.key==pygame.K_SPACE:playing=not playing
                if event.key==pygame.K_RIGHT:roundindex=min(roundindex+1,len(data[['fedavg','fedprox','fedadam'][chosen]])-1)
                if event.key==pygame.K_LEFT:roundindex=max(0,roundindex-1)
        method=['fedavg','fedprox','fedadam'][chosen]
        if playing:
            tick+=dt
            if tick>=1200:
                tick=0;roundindex=(roundindex+1)%len(data[method])
        row=data[method][roundindex];rnum=int(row['round']);f1=float(row['macro_f1']);acc=float(row['accuracy'])
        screen.fill((17,25,39))
        text('Four simulated autonomous vehicles',30,20,big=True)
        text('1 FedAvg   2 FedProx   3 FedAdam   SPACE play/pause   LEFT/RIGHT rounds   ESC quit',30,60,(184,198,218))
        rect(pygame.Rect(28,98,735,524),(30,41,58))
        for x in (155,335,515,695):
            pygame.draw.line(screen,(77,83,97),(x,122),(x,600),4)
            for y in range(130,590,65):pygame.draw.line(screen,(170,176,181),(x,y),(x,y+28),2)
        centers=[(215,215),(570,215),(215,470),(570,470)]
        for i,(x,y) in enumerate(centers):
            # Demonstration graphics are not physical vehicle simulation.
            dx=int(12*math.sin(pygame.time.get_ticks()/900+i))
            rect(pygame.Rect(x+dx-43,y-20,86,40),colors[i])
            rect(pygame.Rect(x+dx-27,y-29,54,15),(200,221,236),3)
            text(f'Vehicle {i+1}',x-43,y+33)
        server=(390,343)
        pygame.draw.circle(screen,(231,239,247),server,62)
        text('FL server',349,333,(20,30,45))
        for i,(x,y) in enumerate(centers):
            t=(pygame.time.get_ticks()%1500)/1500
            xx=int(x+(server[0]-x)*t);yy=int(y+(server[1]-y)*t)
            pygame.draw.circle(screen,colors[i],(xx,yy),7)
        rect(pygame.Rect(788,100,286,522),(30,41,58))
        text('REAL TRAINING LOG',807,120,(177,228,199),big=True)
        text(f'Algorithm: {method}',807,175)
        text(f'Partition: {a.strategy}',807,205)
        text(f'Labeled budget: {a.labels}%',807,235)
        text(f'Round: {rnum} / {len(data[method])}',807,280)
        text(f'Validation macro-F1: {f1:.4f}',807,316)
        text(f'Validation accuracy: {acc:.4f}',807,346)
        rect(pygame.Rect(805,390,243,20),(65,76,91))
        rect(pygame.Rect(805,390,int(243*max(0,min(1,f1))),20),(107,210,171))
        text('Illustrative car and message animation',807,447,(225,174,125))
        text('Not live object detection or training',807,475,(225,174,125))
        text('No raw images exchanged here',807,515,(180,201,217))
        text('Display uses saved validation results',807,543,(180,201,217))
        text('Playback',32,638,(200,215,231));pygame.display.flip()
if __name__=='__main__':main()
