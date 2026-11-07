import { useEffect,useRef,useState } from 'react';
import { Alert,Button,Checkbox,Group,MultiSelect,NumberInput,Select,Slider,Text } from '@mantine/core';
import { Maximize2,Pause,Play,Video } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { api,Camera,Observation,Track,cameraTime,color,observationsAtFrame,shouldSeekVideo,time,useApi,videoPlaybackRate } from './api';

function CameraPanel({camera,dataset,clock,playing,master,boxes,globalId,onMasterFrame,onError}:{camera:Camera;dataset:string;clock:number;playing:boolean;master:boolean;boxes:boolean;globalId:number|null;onMasterFrame:()=>void;onError:(message:string)=>void}){
  const video=useRef<HTMLVideoElement>(null),canvas=useRef<HTMLCanvasElement>(null),container=useRef<HTMLDivElement>(null);
  const samples=useRef<Observation[]>([]),selection=useRef({boxes,globalId}),masterClock=useRef({master,onMasterFrame});selection.current={boxes,globalId};masterClock.current={master,onMasterFrame};
  const [error,setError]=useState(''),[sampleVersion,setSampleVersion]=useState(0);const bucket=Math.floor(clock/2);
  useEffect(()=>{const controller=new AbortController();
    async function load(){let collected:Observation[]=[];let offset=0;while(true){const result=await api<{items:Observation[];total:number}>(`/api/datasets/${dataset}/observations?camera_id=${camera.camera_id}&start=${Math.max(0,bucket*2-.2)}&end=${bucket*2+2.3}&offset=${offset}&limit=10000`,{signal:controller.signal});collected.push(...result.items);offset+=result.items.length;if(offset>=result.total||!result.items.length)break;}samples.current=collected;setSampleVersion(v=>v+1);setError('');}
    if(camera.fps)load().catch(e=>{if(e.name!=='AbortError')setError(e.message)});return()=>controller.abort();
  },[dataset,camera.camera_id,camera.fps,bucket]);
  useEffect(()=>{const v=video.current;if(!v)return;const target=cameraTime(clock,camera);
    const seek=Number.isFinite(v.duration)&&shouldSeekVideo(v.currentTime,target.local,playing);if(seek)v.currentTime=target.local;
    v.playbackRate=playing&&target.active&&!seek?videoPlaybackRate(v.currentTime,target.local):1;
    if(playing&&target.active){if(v.paused)v.play().catch(e=>onError(`Camera ${camera.camera_id}: ${e.message}`));}else v.pause();
  },[clock,playing,camera]);
  useEffect(()=>{
    const v=video.current,c=canvas.current;if(!v||!c)return;let frame=0;
    const draw=(mediaTime=v.currentTime)=>{
      const sourceWidth=v.videoWidth||camera.width||1920,sourceHeight=v.videoHeight||camera.height||1080;
      const pixelRatio=Math.min(window.devicePixelRatio||1,2),width=Math.max(1,Math.round(c.clientWidth*pixelRatio)),height=Math.max(1,Math.round(c.clientHeight*pixelRatio));
      if(c.width!==width)c.width=width;if(c.height!==height)c.height=height;
      const ctx=c.getContext('2d');
      if(ctx){
          ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,width,height);ctx.setTransform(width/sourceWidth,0,0,height/sourceHeight,0,0);
          if(selection.current.boxes&&camera.fps){
            const fid=Math.floor(mediaTime*camera.fps+.001);
            ctx.lineWidth=Math.max(2,sourceWidth/700);ctx.font=`600 ${Math.max(16,sourceWidth/85)}px sans-serif`;
            for(const o of observationsAtFrame(samples.current,fid,camera.fps)){
              if(selection.current.globalId!==null&&o.global_id!==selection.current.globalId)continue;
              const [x1,y1,x2,y2]=o.bbox;
              const chosen=selection.current.globalId!==null;
              if(chosen){ctx.lineWidth=sourceWidth/220;ctx.strokeStyle='#fff';ctx.strokeRect(x1,y1,x2-x1,y2-y1)}
              ctx.lineWidth=Math.max(2,sourceWidth/(chosen?400:700));ctx.strokeStyle=color(o.global_id);ctx.strokeRect(x1,y1,x2-x1,y2-y1);
              const label=o.global_id===null?`Local ${o.local_id} · ID не назначен`:`G${o.global_id} · L${o.local_id}${chosen?' · ВЫБРАН':''}`;
              const h=sourceWidth/70+10;ctx.fillStyle=color(o.global_id);
              ctx.fillRect(x1,Math.max(0,y1-h),ctx.measureText(label).width+12,h);
              ctx.fillStyle='#10151e';ctx.fillText(label,x1+6,Math.max(h,y1)-6);
            }
          }
      }
    };
    const redraw=()=>draw(),next:VideoFrameRequestCallback=(_,metadata)=>{draw(metadata.mediaTime);if(masterClock.current.master)masterClock.current.onMasterFrame();frame=v.requestVideoFrameCallback(next)};
    if(!playing)draw();
    if(playing&&'requestVideoFrameCallback' in v)frame=v.requestVideoFrameCallback(next);else v.addEventListener('timeupdate',redraw);
    window.addEventListener('resize',redraw);
    return()=>{if(frame)v.cancelVideoFrameCallback(frame);v.removeEventListener('timeupdate',redraw);window.removeEventListener('resize',redraw)};
  },[camera,playing,boxes,globalId,sampleVersion]);
  const active=cameraTime(clock,camera).active;
  return <div className="camera-panel" ref={container}><div className="camera-heading"><span><i/> {camera.name||`Camera ${camera.camera_id}`}</span><span>{camera.fps??'—'} FPS <button aria-label={`Увеличить камеру ${camera.camera_id}`} onClick={()=>container.current?.requestFullscreen()}><Maximize2 size={16}/></button></span></div>
    <div className="video-stage" style={{aspectRatio:`${camera.width||16}/${camera.height||9}`}}>
      {camera.video_url?<><video ref={video} src={camera.video_url} muted playsInline preload="metadata" onLoadedMetadata={e=>{e.currentTarget.currentTime=cameraTime(clock,camera).local}} onError={()=>setError('Браузер не может воспроизвести кодек. Для просмотра используйте MP4 H.264; декодирование OpenCV проверяется отдельно.')} /><canvas ref={canvas} data-testid={`overlay-${camera.camera_id}`}/>{!active&&<div className="video-note">Вне временного диапазона камеры</div>}</>:<div className="video-note"><Video size={32}/> Видео не подключено</div>}
    </div><div className="camera-footer">CAM {String(camera.camera_id).padStart(2,'0')}<span>{camera.width??'—'} × {camera.height??'—'} · смещение {camera.offset??0} с</span></div>
    {!camera.fps&&<Text size="xs" c="yellow">FPS неизвестен — привязка рамок ко времени недоступна</Text>}{error&&<Alert color="red">{error}</Alert>}</div>
}

export function Viewer({id}:{id:string}){
  const {data:cameras,error}=useApi<Camera[]>(`/api/datasets/${id}/cameras`);const [params]=useSearchParams();
  const {data:details}=useApi(`/api/datasets/${id}`);
  const [clock,setClock]=useState(Number(params.get('time')||0)),[playing,setPlaying]=useState(false),[masterStarted,setMasterStarted]=useState(false),[boxes,setBoxes]=useState(true),[selected,setSelected]=useState<string[]>([]),[columns,setColumns]=useState('3'),[failure,setFailure]=useState('');
  const [gid,setGid]=useState<number|string>(params.get('global')??'');const clockRef=useRef(clock);clockRef.current=clock;
  const max=Math.max(1,...(cameras||[]).map(c=>(c.duration??0)+(c.offset??0))),min=Math.min(0,...(cameras||[]).map(c=>c.offset??0));
  useEffect(()=>{if(!playing||!masterStarted)return;const timer=setInterval(()=>{clockRef.current=Math.min(clockRef.current+.1,max);setClock(clockRef.current);if(clockRef.current>=max)setPlaying(false)},100);return()=>clearInterval(timer)},[playing,masterStarted,max]);
  useEffect(()=>{const next=Number(params.get('time')||0);setPlaying(false);setMasterStarted(false);setGid(params.get('global')??'');clockRef.current=next;setClock(next);},[id,params]);
  async function chooseGlobal(value:number|string){setGid(value);if(value==='')return;try{const track=await api<Track>(`/api/datasets/${id}/tracks/${Number(value)}`);if(track.start_time!==null){setPlaying(false);setMasterStarted(false);clockRef.current=track.start_time;setClock(track.start_time)}}catch(e){setFailure((e as Error).message)}}
  const shown=(cameras||[]).filter(c=>!selected.length||selected.includes(String(c.camera_id)));
  const master=shown.find(c=>c.video_url&&cameraTime(clock,c).active)?.camera_id;
  function startFromMaster(){if(!masterStarted)setMasterStarted(true)}
  return <><div className="page-title"><div className="eyebrow">СИНХРОННОЕ НАБЛЮДЕНИЕ</div><h1>Камеры</h1><p>Единое время и идентификаторы из результатов алгоритма</p></div>
    {(error||failure)&&<Alert color="red" mb="md">{error||failure}</Alert>}
    <Group className="toolbar" justify="space-between"><Group><MultiSelect aria-label="Фильтр камер" placeholder="Все камеры" data={(cameras||[]).map(c=>({value:String(c.camera_id),label:c.name}))} value={selected} onChange={setSelected}/><NumberInput aria-label="Выбрать Global ID" placeholder="Все Global ID" min={0} value={gid} onChange={chooseGlobal} w={160}/><Checkbox label="Bounding boxes" checked={boxes} onChange={e=>setBoxes(e.currentTarget.checked)}/>{gid!==''&&<Text size="sm" c="blue">Выбран Global {gid} · видео с начала трека</Text>}</Group><Select aria-label="Сетка камер" w={130} value={columns} onChange={v=>setColumns(v||'2')} data={[{value:'2',label:'2 колонки'},{value:'3',label:'3 колонки'}]}/></Group>
    <Text size="xs" c="dimmed" mb="sm">Наблюдения в результате: {time(details?.summary.result_start)} — {time(details?.summary.result_end)}. Полная длительность видео может быть больше.</Text>
    <div className="camera-grid" style={{gridTemplateColumns:`repeat(${columns},minmax(0,1fr))`}}>{shown.map(c=><CameraPanel key={c.camera_id} camera={c} dataset={id} clock={clock} playing={playing&&(c.camera_id===master||masterStarted)} master={c.camera_id===master} boxes={boxes} globalId={gid===''?null:Number(gid)} onMasterFrame={startFromMaster} onError={message=>{setPlaying(false);setFailure(message)}}/>)}</div>
    {!cameras?.length&&<div className="empty">Подключите видео или загрузите результаты с Camera ID</div>}
    <div className="transport"><Button onClick={()=>{if(playing)setPlaying(false);else{setMasterStarted(false);setPlaying(true)}}} disabled={master===undefined} leftSection={playing?<Pause size={16}/>:<Play size={16}/>}>{playing?'Пауза':'Воспроизвести'}</Button><Text ff="monospace">{time(clock)}</Text><Slider aria-label="Общий таймлайн" min={min} max={max} step={.0333333} value={clock} onChange={v=>{setPlaying(false);setMasterStarted(false);clockRef.current=v;setClock(v)}} label={time} style={{flex:1}}/><Text c="dimmed" ff="monospace">{time(max)}</Text></div>
  </>;
}
