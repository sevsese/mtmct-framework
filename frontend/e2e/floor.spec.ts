import {test,expect} from '@playwright/test';

test('manual floor mapping and selecting a person',async({page,request})=>{
  const response=await request.post('/api/datasets/import',{data:{relative_path:'runs/nvidia_branch_1920',name:'DEMO — floor map UI calibration test'}});
  expect(response.ok()).toBeTruthy();const {id}=await response.json();
  await page.addInitScript(id=>localStorage.setItem('mtmct-dataset',id),id);
  await page.goto('/floorplan');
  // Synthetic blank plan, explicitly a UI test, not a physical calibration.
  await page.locator('input[type=file]').setInputFiles({name:'DEMO.png',mimeType:'image/png',buffer:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAI0lEQVQIHW3BAQEAAAABIP6PzgFV5CnyFHmKPEWeIk+Rp8gzVoYX8WHqMxQAAAAASUVORK5CYII=','base64')});
  await page.getByRole('button',{name:'Загрузить план',exact:true}).click();
  await expect(page.getByAltText('Схематический план помещения')).toBeVisible();
  await page.getByRole('textbox',{name:'Камера',exact:true}).click();
  await page.getByRole('option',{name:'Camera 3',exact:true}).click();
  await page.getByRole('button',{name:'Привязать к плану'}).click();
  const frame=page.locator('.calibration-frame'),plan=page.locator('.trajectory-map');
  await expect(page.getByAltText('Кадр для привязки к плану')).toBeVisible();
  await expect.poll(()=>page.getByAltText('Кадр для привязки к плану').evaluate((img:HTMLImageElement)=>img.naturalWidth)).toBeGreaterThan(0);
  for(const [x,y] of [[.01,.01],[.99,.01],[.99,.99],[.01,.99]]){
    const a=await frame.boundingBox(),b=await plan.boundingBox();
    await frame.click({position:{x:x*a!.width,y:y*a!.height}});
    await plan.click({position:{x:x*b!.width,y:y*b!.height}});
  }
  await page.getByRole('button',{name:'Сохранить привязку'}).click();
  await expect(page.getByRole('button',{name:'Воспроизвести карту'})).toBeVisible();
  await page.goto('/floorplan?time=1');
  await expect(page.locator('.person-pin').first()).toBeVisible();
  await page.locator('.person-pin').first().click();
  await expect(page.getByRole('textbox',{name:'Global ID на плане'})).not.toHaveValue('');
  await expect(page.locator('polyline').first()).toBeVisible();
  await page.getByRole('link',{name:'Открыть этот момент на видео'}).click();
  await page.getByRole('button',{name:'Отдельные камеры'}).click();
  await expect(page.locator('video')).toHaveCount(2);
});
