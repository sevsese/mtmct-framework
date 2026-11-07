import {test,expect} from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
const shots=path.resolve('../docs/web-screenshots');
test('real NVIDIA results, all routes, synchronized video and overlays',async({page,request})=>{
  const response=await request.post('http://127.0.0.1:8000/api/datasets/import',{data:{relative_path:'runs/adaptive_final_development/full',name:'Warehouse_000 · adaptive · 1920'}});
  expect(response.ok()).toBeTruthy();const {id}=await response.json();
  await page.addInitScript(id=>localStorage.setItem('mtmct-dataset',id),id);
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  fs.mkdirSync(shots,{recursive:true});
  await page.goto('/');await expect(page.getByRole('heading',{name:'Warehouse_000 · adaptive · 1920'})).toBeVisible();
  await expect(page.getByText('Глобальные треки',{exact:true}).last()).toBeVisible();
  await page.screenshot({path:path.join(shots,'dashboard-1366.png'),fullPage:true});
  await page.getByRole('link',{name:'Камеры',exact:true}).click();
  await page.getByRole('button',{name:'Отдельные камеры'}).click();
  await expect(page.locator('video')).toHaveCount(2);
  await expect.poll(()=>page.locator('video').evaluateAll(videos=>videos.every(v=>(v as HTMLVideoElement).readyState>=2))).toBeTruthy();
  await expect.poll(()=>page.locator('canvas').first().evaluate(c=>{
    const canvas=c as HTMLCanvasElement;const pixels=canvas.getContext('2d')!.getImageData(0,0,canvas.width,canvas.height).data;let count=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i])count++;return count;
  })).toBeGreaterThan(100);
  await page.screenshot({path:path.join(shots,'viewer-1366.png'),fullPage:true});
  await page.getByRole('button',{name:'Воспроизвести',exact:true}).click();
  await expect.poll(()=>page.locator('video').first().evaluate(v=>(v as HTMLVideoElement).currentTime)).toBeGreaterThan(1);
  await page.getByRole('button',{name:'Пауза',exact:true}).click();
  const times=await page.locator('video').evaluateAll(v=>v.map(x=>(x as HTMLVideoElement).currentTime));
  expect(Math.abs(times[0]-times[1])).toBeLessThan(.2);
  await page.getByRole('checkbox',{name:'Bounding boxes'}).uncheck();
  await expect.poll(()=>page.locator('canvas').first().evaluate(c=>{
    const canvas=c as HTMLCanvasElement;return canvas.getContext('2d')!.getImageData(0,0,canvas.width,canvas.height).data.some((v,i)=>i%4===3&&v>0);
  })).toBeFalsy();
  await page.getByRole('link',{name:'Глобальные треки',exact:true}).click();
  await expect(page.locator('tbody tr').first()).toBeVisible();
  await page.screenshot({path:path.join(shots,'tracks-1366.png'),fullPage:true});
  const first=page.locator('tbody a').first();const gid=(await first.innerText()).replace('Global ','');
  await page.getByRole('textbox',{name:'Поиск Global ID'}).fill(gid);await expect(page.locator('tbody tr').first()).toBeVisible();
  await page.locator('tbody a').first().click();await expect(page.getByRole('heading',{name:`Global ID ${gid}`})).toBeVisible();
  await page.getByRole('link',{name:'Показать на видео'}).click();
  await page.getByRole('button',{name:'Отдельные камеры'}).click();await expect(page.locator('video')).toHaveCount(2);
  for(const [label,title] of [['Аналитика','Аналитика'],['Настройки','Настройки результата'],['План помещения','План помещения']]){await page.getByRole('link',{name:label,exact:true}).click();await expect(page.getByRole('heading',{name:title,exact:true})).toBeVisible();}
  await expect(page.getByText('План пока не загружен')).toBeVisible();
  await page.getByRole('link',{name:'Аналитика',exact:true}).click();await expect(page.getByText(/global-2d-idf1-v1/)).toBeVisible();
  await page.setViewportSize({width:1920,height:1080});await page.screenshot({path:path.join(shots,'analytics-1920.png'),fullPage:true});
  await page.getByRole('link',{name:'Новый набор',exact:true}).click();await expect(page.getByRole('heading',{name:'Новый набор',exact:true})).toBeVisible();
  expect(errors).toEqual([]);
});

test('CSV upload through UI and malformed file errors',async({page})=>{
  await page.goto('/upload');await page.getByRole('textbox',{name:'Название набора'}).fill('DEMO — UI upload test');
  const files=page.locator('input[type=file]');
  await files.first().setInputFiles({name:'global_tracks_2d.csv',mimeType:'text/csv',buffer:Buffer.from('bad,header\n1,2\n')});
  await page.getByRole('button',{name:'Загрузить набор'}).click();await expect(page.getByText(/отсутствуют столбцы/)).toBeVisible();
  await files.first().setInputFiles({name:'global_tracks_2d.csv',mimeType:'text/csv',buffer:Buffer.from('camera_id,frame_id,local_id,x1,y1,x2,y2,global_id\n3,0,1,0,0,10,20,99\n')});
  await page.getByRole('button',{name:'Загрузить набор'}).click();await expect(page.getByRole('heading',{name:'DEMO — UI upload test'})).toBeVisible();
  await page.getByRole('link',{name:'Глобальные треки',exact:true}).click();await expect(page.getByRole('link',{name:'Global 99',exact:true})).toBeVisible();
});

test('real NVIDIA video upload and GPU processing through UI',async({page,request})=>{
  test.skip(process.env.WEB_GPU_TEST!=='1','Opt in with WEB_GPU_TEST=1; requires local models and NVIDIA excerpts');
  test.setTimeout(180000);
  await page.goto('/upload');
  await page.getByRole('textbox',{name:'Название набора'}).fill('NVIDIA — проверка UI, 12 кадров');
  await page.getByRole('textbox',{name:'Режим загрузки'}).click();
  await page.getByRole('option',{name:'Исходные видео → обработка'}).click();
  await page.locator('input[type=file]').setInputFiles([3,15].map(c=>path.resolve(`../runs/web_pipeline_smoke/Camera_${String(c).padStart(4,'0')}.mp4`)));
  await page.getByRole('button',{name:'Загрузить набор',exact:true}).click();
  await expect(page.getByRole('heading',{name:'NVIDIA — проверка UI, 12 кадров'})).toBeVisible();
  await page.getByRole('textbox',{name:'Максимум кадров (0 — всё видео)'}).fill('12');
  await page.getByRole('button',{name:'Запустить обработку',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Обработка завершена',exact:true})).toBeVisible({timeout:150000});
  const id=await page.evaluate(()=>localStorage.getItem('mtmct-dataset'));
  const result=await (await request.get(`/api/datasets/${id}`)).json();
  const performance=await (await request.get(`/api/datasets/${id}/performance`)).json();
  expect(result.summary.global_tracks).toBeGreaterThan(0);
  expect(result.summary.observations).toBeGreaterThan(0);
  expect(performance.cameras.map((c:any)=>c.processed_frames)).toEqual([12,12]);
  await page.screenshot({path:path.join(shots,'processing-result-1366.png'),fullPage:true});
});

test('backend unavailable is explained in the UI',async({page})=>{
  await page.route('**/api/**',route=>route.abort());
  await page.goto('/');
  await expect(page.getByText('Backend недоступен',{exact:true})).toBeVisible();
  await expect(page.getByText(/Проверьте запуск сервера/)).toBeVisible();
});
