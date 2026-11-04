// Real-backend smoke for the owner split (not mocked). Needs: backend with MAIL_BACKEND=log, OWNER_EMAILS=owner@test.local,
// LOG=<backend log file>, and vite at BASE with API_TARGET pointing at it. Run:
//   LOG=/path/be.log BASE=http://127.0.0.1:5431 API=http://127.0.0.1:8030 node tests/e2e-owner-realbackend.mjs
// Prints findings (leak scan per viewer, nav flicker on reload, 390px overflow per owner section). Exits 1 on a leak or overflow.
import { chromium } from 'playwright'
import fs from 'fs'
const LOGF=process.env.LOG; process.exitCode=0
const B=process.env.BASE||'http://127.0.0.1:5431'; const API=process.env.API||'http://127.0.0.1:8030'
const link=async e=>{await fetch(''+API+'/api/auth/request-link',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({email:e})});await new Promise(r=>setTimeout(r,500));return [...fs.readFileSync(LOGF,'utf8').matchAll(new RegExp('link for '+e.replace('.','\\.')+'.*?token=([\\w-]+)','g'))].pop()[1]}
const br=await chromium.launch()
const mk=async(vp={width:1280,height:900})=>{const ctx=await br.newContext({viewport:vp});const p=await ctx.newPage();p.errs=[];p.on('console',m=>m.type()==='error'&&!/Failed to load resource/.test(m.text())&&p.errs.push(m.text()));p.on('pageerror',e=>p.errs.push(e.message));return p}
const signIn=async(p,e)=>{const t=await link(e);await p.goto(B+'/#/verify?token='+t);await p.waitForTimeout(2500)}
const LEAK=/Save changes|Link videos|Add photo|Add connection|Delete place|Inspect|Rebuild|Run next|Resolve|@test\.local|@example\.com|Moderation queue|Block/i
const leak=async(p,label)=>{for(const h of ['navigate','map','record','build','owner']){await p.goto(B+'/#/'+h);await p.reload();await p.waitForTimeout(1500);const html=await p.evaluate(()=>document.body.innerText+' '+[...document.querySelectorAll('button,input,a,select,textarea')].map(e=>e.innerText+' '+(e.getAttribute('aria-label')||'')+' '+(e.placeholder||'')).join(' '));const m=html.match(LEAK);console.log(label,h,m?'LEAK '+m[0]:'clean');if(m)process.exitCode=1}}
const v=await mk(); await leak(v,'visitor'); console.log(v.errs)
const w=await mk(); await signIn(w,'walker2@test.local'); await leak(w,'contrib'); console.log(w.errs)
const o=await mk(); await signIn(o,'owner@test.local')
for(const h of ['navigate','map','owner']){await o.goto(B+'/#/'+h);const t0=Date.now();o.reload();const seen=[];let prev='';for(let i=0;i<30;i++){const s=await o.evaluate(()=>[...document.querySelectorAll('nav button, nav a, [role=tab]')].map(e=>e.innerText.trim()).filter(Boolean).join(',')).catch(()=>'x');if(s!==prev){seen.push((Date.now()-t0)+'ms '+s);prev=s}await o.waitForTimeout(40)}console.log('flicker',h,seen)}
await o.setViewportSize({width:390,height:800});await o.goto(B+'/#/owner?s=users');await o.reload();await o.waitForTimeout(1800)
const ovf=await o.evaluate(()=>document.documentElement.scrollWidth>innerWidth);console.log('390 users page overflow',ovf);if(ovf)process.exitCode=1
console.log(await o.evaluate(()=>{const o=[];document.querySelectorAll('body *').forEach(e=>{const r=e.getBoundingClientRect();if(r.right>391)o.push(e.tagName+'.'+e.className+' '+Math.round(r.right))});return o.slice(0,8)}))
await br.close()
