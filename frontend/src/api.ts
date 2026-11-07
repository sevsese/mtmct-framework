import { useEffect, useState } from 'react';
export type Camera = {camera_id:number;name:string;offset:number;fps:number|null;width?:number;height?:number;duration?:number;video_url:string|null};
export type Observation = {camera_id:number;frame_id:number;local_id:number;global_id:number|null;timestamp:number|null;bbox:number[];confidence:number|null};
export type Track = {global_id:number;start_time:number|null;end_time:number|null;duration:number|null;cameras:number[];observations_count:number|null;status:string|null;local_tracklets:{camera_id:number;local_id:number}[]};
export type Dataset = {id:string;name:string;mode:string;status:string;warnings:string[];files:string[];created_at:string};
export async function api<T=any>(path:string,init?:RequestInit):Promise<T>{
  let response:Response;
  try {response=await fetch(path,init);} catch(e){if((e as Error).name==='AbortError')throw e;throw new Error('Backend недоступен. Проверьте запуск сервера на порту 8000.');}
  if(!response.ok){const body=await response.json().catch(()=>({detail:response.statusText}));throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail));}
  return response.json();
}
export const post=(path:string,body:unknown)=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
export function useApi<T=any>(url:string|null,revision=0,keepPrevious=false){
  const [data,setData]=useState<T|null>(null),[error,setError]=useState('');
  useEffect(()=>{if(!keepPrevious||!url)setData(null);setError('');if(!url)return;const abort=new AbortController();api<T>(url,{signal:abort.signal}).then(setData).catch(e=>{if(e.name!=='AbortError')setError(e.message)});return()=>abort.abort()},[url,revision,keepPrevious]);
  return {data,error};
}
export const color=(id:number|null)=>id===null?'#aab4c5':`hsl(${((id*137.508)%360+360)%360} 78% 67%)`;
export const time=(seconds:number|null|undefined)=>seconds==null?'Нет данных':`${seconds<0?'-':''}${Math.floor(Math.abs(seconds)/60).toString().padStart(2,'0')}:${(Math.abs(seconds)%60).toFixed(1).padStart(4,'0')}`;
export function cameraTime(globalTime:number,camera:Pick<Camera,'offset'|'duration'>){const local=globalTime-camera.offset;return {local:Math.max(0,Math.min(local,camera.duration??Infinity)),active:local>=0&&(camera.duration==null||local<camera.duration)}}
export const shouldSeekVideo=(current:number,target:number,playing:boolean)=>Math.abs(current-target)>(playing?.75:.04);
export function observationsAtFrame(samples:Observation[],frame:number,fps:number){
  const limit=Math.max(1,Math.ceil(fps*.12)),tracks=new Map<number,{before?:Observation;after?:Observation}>();
  for(const sample of samples){const delta=sample.frame_id-frame;if(Math.abs(delta)>limit)continue;const pair=tracks.get(sample.local_id)||{};if(delta<=0&&(!pair.before||sample.frame_id>pair.before.frame_id))pair.before=sample;if(delta>=0&&(!pair.after||sample.frame_id<pair.after.frame_id))pair.after=sample;tracks.set(sample.local_id,pair)}
  return [...tracks.values()].map(({before,after})=>{if(before&&after&&before.frame_id!==after.frame_id&&before.global_id===after.global_id){const ratio=(frame-before.frame_id)/(after.frame_id-before.frame_id);return {...before,frame_id:frame,bbox:before.bbox.map((value,index)=>value+(after.bbox[index]-value)*ratio)}}return !before?after:!after?before:frame-before.frame_id<=after.frame_id-frame?before:after}).filter((sample):sample is Observation=>!!sample);
}
