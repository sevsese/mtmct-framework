import { defineConfig } from '@playwright/test';
export default defineConfig({testDir:'./e2e',workers:1,timeout:60000,
  use:{baseURL:process.env.WEB_URL||'http://127.0.0.1:8000',channel:process.env.WEB_BROWSER||'msedge',headless:true,viewport:{width:1366,height:768}},
  reporter:'list'});
