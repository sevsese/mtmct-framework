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
export function useApi<T=any>(url:string|null,revision=0){
  const [data,setData]=useState<T|null>(null),[error,setError]=useState('');
  useEffect(()=>{setData(null);setError('');if(!url)return;const abort=new AbortController();api<T>(url,{signal:abort.signal}).then(setData).catch(e=>{if(e.name!=='AbortError')setError(e.message)});return()=>abort.abort()},[url,revision]);
  return {data,error};
}
export const color=(id:number|null)=>id===null?'#aab4c5':`hsl(${((id*137.508)%360+360)%360} 78% 67%)`;
export const time=(seconds:number|null|undefined)=>seconds==null?'Нет данных':`${seconds<0?'-':''}${Math.floor(Math.abs(seconds)/60).toString().padStart(2,'0')}:${(Math.abs(seconds)%60).toFixed(1).padStart(4,'0')}`;
export function cameraTime(globalTime:number,camera:Pick<Camera,'offset'|'duration'>){const local=globalTime-camera.offset;return {local:Math.max(0,Math.min(local,camera.duration??Infinity)),active:local>=0&&(camera.duration==null||local<camera.duration)}}
