import { describe,it,expect } from 'vitest';
import { cameraTime,color,observationsAtFrame,shouldSeekVideo,time } from './api';
describe('Video time and identity presentation',()=>{
  it('applies offset once and clamps recordings with different durations',()=>{
    expect(cameraTime(12,{offset:2,duration:20})).toEqual({local:10,active:true});
    expect(cameraTime(1,{offset:2,duration:20})).toEqual({local:0,active:false});
    expect(cameraTime(50,{offset:2,duration:20})).toEqual({local:20,active:false});
    expect(cameraTime(1,{offset:-3,duration:10})).toEqual({local:4,active:true});
  });
  it('uses the same global color on all cameras and a neutral local fallback',()=>{
    expect(color(7)).toBe(color(7));expect(color(7)).not.toBe(color(8));expect(color(null)).toBe('#aab4c5');
    expect(time(null)).toBe('Нет данных');expect(time(61.5)).toBe('01:01.5');
  });
  it('does not repeatedly reset the decoder for small playback drift',()=>{
    expect(shouldSeekVideo(10,10.3,true)).toBe(false);
    expect(shouldSeekVideo(10,11,true)).toBe(true);
    expect(shouldSeekVideo(10,10.1,false)).toBe(true);
  });
  it('interpolates sparse tracking results without holding stale boxes',()=>{
    const sample=(frame_id:number,x:number)=>({camera_id:0,frame_id,local_id:1,global_id:7,timestamp:frame_id/25,bbox:[x,0,x+10,20],confidence:.9});
    expect(observationsAtFrame([sample(2,20),sample(4,40)],3,25)[0].bbox).toEqual([30,0,40,20]);
    expect(observationsAtFrame([sample(2,20)],6,25)).toEqual([]);
  });
});
