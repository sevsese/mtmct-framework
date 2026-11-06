import { describe,it,expect } from 'vitest';
import { cameraTime,color,time } from './api';
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
});
