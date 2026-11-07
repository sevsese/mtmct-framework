import {useEffect,useState} from 'react';
import {Alert,Button,FileInput,Group,NumberInput,Select,Slider,Text} from '@mantine/core';
import {Link,useSearchParams} from 'react-router-dom';
import {api,Camera,color,time,useApi} from './api';

type Pair={video:[number,number];plan:[number,number]};
type Point={camera_id:number;local_id:number;global_id:number|null;timestamp:number;x:number;y:number};
const xy=(e:React.MouseEvent<HTMLElement>):[number,number]=>{const r=e.currentTarget.getBoundingClientRect();return [(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]};

export function FloorPlan({id}:{id:string}){
  const [params]=useSearchParams();
  const [revision,setRevision]=useState(0),[file,setFile]=useState<File|null>(null),[camera,setCamera]=useState<string|null>(null);
  const [error,setError]=useState(''),[editing,setEditing]=useState(false),[pairs,setPairs]=useState<Pair[]>([]),[pending,setPending]=useState<[number,number]|null>(null);
  const [frameTime,setFrameTime]=useState<number|string>(0),[clock,setClock]=useState(Number(params.get('time')||0)),[gid,setGid]=useState<number|string>(params.get('global')||'');
  const [playing,setPlaying]=useState(false);
  const {data,error:loadError}=useApi(`/api/datasets/${id}/floorplan`,revision);
  const {data:cameras}=useApi<Camera[]>(`/api/datasets/${id}/cameras`);
  const {data:details}=useApi(`/api/datasets/${id}`);
  const end=Math.max(1,details?.summary?.result_end??Math.max(...(cameras||[]).map(c=>c.duration||0),1));
  const {data:routes,error:routeError}=useApi<{points:Point[];truncated:boolean;warning:string}>(data?.calibrated&&!editing?`/api/datasets/${id}/floorplan/trajectories?start=${Math.max(0,clock-15)}&end=${clock}${gid===''?'':`&global_id=${gid}`}`:null,revision,true);
  useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setClock(t=>{if(t+.2>=end){setPlaying(false);return end}return t+.2}),200);return()=>clearInterval(timer)},[playing,end]);
  useEffect(()=>{setPairs([]);setPending(null)},[camera]);
  async function upload(){if(!file)return;try{const body=new FormData();body.append('file',file);await api(`/api/datasets/${id}/floorplan`,{method:'POST',body});setRevision(r=>r+1);setError('')}catch(e){setError((e as Error).message)}}
  async function save(){if(!camera)return;try{await api(`/api/datasets/${id}/floorplan/calibration/${camera}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(pairs)});setRevision(r=>r+1);setEditing(false);setError('')}catch(e){setError((e as Error).message)}}
  async function mapClick(e:React.MouseEvent<HTMLDivElement>){
    if(!camera)return;const [x,y]=xy(e);
    if(editing){if(pending){setPairs(p=>[...p,{video:pending,plan:[x,y]}]);setPending(null)}return}
    try{await api(`/api/datasets/${id}/floorplan`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify([...(data?.markers||[]).filter((m:any)=>m.camera_id!==Number(camera)),{camera_id:Number(camera),x,y}])});setRevision(r=>r+1)}catch(e){setError((e as Error).message)}
  }
  const groups=new Map<string,Point[]>();
  for(const p of routes?.points||[]){if(p.timestamp>clock||p.timestamp<clock-15||(gid!==''&&p.global_id!==Number(gid)))continue;const key=`${p.camera_id}:${p.local_id}:${p.global_id}`;if(!groups.has(key))groups.set(key,[]);groups.get(key)!.push(p)}
  const segments:Point[][]=[];const heads:Point[]=[];
  for(const points of groups.values()){
    let segment:Point[]=[];
    for(const p of points){if(segment.length&&p.timestamp-segment[segment.length-1].timestamp>.5){segments.push(segment);segment=[]}segment.push(p)}
    if(segment.length)segments.push(segment);
    const last=points[points.length-1];if(clock-last.timestamp<=.25)heads.push(last);
  }
  return <>
    <div className="page-title"><div className="eyebrow">ПРОЕКЦИЯ ТРЕКОВ НА ПОЛ</div><h1>План помещения</h1><p>Выбор Global ID, положение людей и история за последние 15 секунд</p></div>
    {(error||loadError||routeError)&&<Alert color="red" mb="md">{error||loadError||routeError}</Alert>}
    <Alert color="yellow" mb="md">{data?.calibrated?'Ручная привязка: приблизительные координаты на плане, не метры и не точная 3D-реконструкция. Линии не соединяют разрывы и разные камеры.':'Камеры можно расставить на схеме. Для координат людей привяжите минимум 4 точки пола каждой камеры к плану.'}</Alert>
    <Group align="end" mb="md"><FileInput label="План PNG / JPG" description="Новый план сбрасывает привязки" accept="image/png,image/jpeg" value={file} onChange={setFile}/><Button disabled={!file} onClick={upload}>Загрузить план</Button>
      <Select label="Камера" placeholder="Выберите камеру" data={(cameras||[]).map(c=>({value:String(c.camera_id),label:`${c.name}${data?.calibrations?.[c.camera_id]?' · привязана':''}`}))} value={camera} onChange={setCamera}/>
      <Button variant="light" disabled={!camera||!data?.available} onClick={()=>{setEditing(!editing);setPlaying(false);setPairs([]);setPending(null)}}>{editing?'Закрыть привязку':'Привязать к плану'}</Button>
    </Group>
    {editing&&<section className="panel"><h2>Соответствия точек пола · Camera {camera}</h2><Text size="sm" mb="md">Нажмите точку пола на кадре, затем ту же точку на плане. Выбирайте углы пола и пороги, не столы или головы. Покройте область движения: за пределами выбранного многоугольника проекция не строится.</Text>
      <Group mb="md"><NumberInput label="Время кадра камеры, с" min={0} value={frameTime} onChange={setFrameTime}/><Text>{pending?'Теперь отметьте ту же точку на плане':`Пар точек: ${pairs.length}. Следующая точка — на видео.`}</Text><Button disabled={pairs.length<4||pending!==null} onClick={save}>Сохранить привязку</Button><Button variant="subtle" onClick={()=>{setPending(null);setPairs(p=>p.slice(0,-1))}}>Отменить точку</Button></Group>
      <div className="calibration-frame" onClick={e=>{if(pairs.length<20)setPending(xy(e))}}><img alt="Кадр для привязки к плану" src={`/api/datasets/${id}/frame/${camera}?time=${Number(frameTime)||0}`}/>
        {[...pairs.map(p=>p.video),...(pending?[pending]:[])].map(([x,y],i)=><b key={i} style={{left:`${x*100}%`,top:`${y*100}%`}}>{i+1}</b>)}</div>
    </section>}
    {!editing&&data?.calibrated&&<Group className="toolbar"><Button onClick={()=>setPlaying(p=>!p)}>{playing?'Пауза карты':'Воспроизвести карту'}</Button><NumberInput aria-label="Global ID на плане" placeholder="Все Global ID" min={0} value={gid} onChange={setGid}/><Text>{time(clock)}</Text><Slider aria-label="Время на плане" min={0} max={end} step={.04} value={clock} onChange={t=>{setPlaying(false);setClock(t)}} style={{flex:1}}/><Button component={Link} variant="light" to={`/viewer?time=${clock}${gid===''?'':`&global=${gid}`}`}>Открыть этот момент на видео</Button></Group>}
    {data?.available?<div className="floorplan trajectory-map" onClick={mapClick}>
      <img alt="Схематический план помещения" src={`/api/datasets/${id}/floorplan/image?v=${revision}`}/>
      <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label="Маршруты людей">
        {!editing&&segments.map((s,i)=><polyline key={i} points={s.map(p=>`${p.x*1000},${p.y*1000}`).join(' ')} fill="none" stroke={color(s[0].global_id)} strokeWidth="2" vectorEffect="non-scaling-stroke"/>)}
      </svg>
      {!editing&&heads.map(p=><button className="person-pin" key={`${p.camera_id}:${p.local_id}`} style={{left:`${p.x*100}%`,top:`${p.y*100}%`,borderColor:color(p.global_id)}} onClick={e=>{e.stopPropagation();if(p.global_id!==null)setGid(p.global_id)}} title={`Camera ${p.camera_id}, Local ${p.local_id}`}><i style={{background:color(p.global_id)}}/>{p.global_id===null?`L${p.local_id}`:`G${p.global_id}`} · C{p.camera_id}</button>)}
      {editing?pairs.map((p,i)=><b className="map-pair" key={i} style={{left:`${p.plan[0]*100}%`,top:`${p.plan[1]*100}%`}}>{i+1}</b>):data.markers.map((m:any)=><span key={m.camera_id} style={{left:`${100*m.x}%`,top:`${100*m.y}%`}}>C{m.camera_id}</span>)}
    </div>:<section className="panel empty"><h2>План пока не загружен</h2><p>Загрузите схему, затем привяжите камеры.</p></section>}
    {routes?.truncated&&<Alert color="yellow">Слишком много точек: выберите один Global ID.</Alert>}
    {!editing&&data?.available&&<Text size="sm" c="dimmed" mt="md">Выбранная камера размещается нажатием на план. Люди показаны только при наличии наблюдения и привязки; отсутствующие участки маршрута не восстанавливаются догадкой.</Text>}
  </>;
}
