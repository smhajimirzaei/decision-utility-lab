// Verify only the dedicated headless browser on9231, and close it afterward.
const fs = require("node:fs"),
  path = require("node:path");
async function main() {
  const root = process.argv[2],
    full = JSON.parse(fs.readFileSync(path.join(root, "report.json"), "utf8"));
  const gold = full.decisions.map((r) => [
    r.cohort,
    r.episode_id,
    r.step,
    r.bank,
    r.size,
    r.selector,
    r.chosen,
    ...["success", "base", "gain", "help", "harm", "regret", "opportunity"].map(
      (k) => r[k],
    ),
  ]);
  const target = path.basename(root) + "/index.html";
  let pages;
  for (let i = 0; i < 50; i++) {
    try {
      pages = await (await fetch("http://127.0.0.1:9231/json/list")).json();
      if (pages.some((p) => p.url.includes(target))) break;
    } catch {}
    await new Promise((r) => setTimeout(r, 200));
  }
  const page = pages.find((p) => p.url.includes(target));
  if (!page) throw Error("Owned audit page unavailable");
  const socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = reject;
  });
  let id = 0;
  const pending = new Map();
  socket.onmessage = (e) => {
    let m = JSON.parse(e.data);
    if (pending.has(m.id)) {
      let q = pending.get(m.id);
      pending.delete(m.id);
      m.error ? q.reject(Error(m.error.message)) : q.resolve(m.result);
    }
  };
  const call = (method, params = {}) =>
    new Promise((resolve, reject) => {
      let n = ++id;
      pending.set(n, { resolve, reject });
      socket.send(JSON.stringify({ id: n, method, params }));
    });
  const evaluate = async (expression) => {
    let r = await call("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  try {
    await call("Runtime.enable");
    await call("Page.enable");
    let ready = false;
    for (let i = 0; i < 100; i++) {
      if (await evaluate("typeof window.DECISION_AUDIT==='object'")) {
        ready = true;
        break;
      }
      await new Promise((r) => setTimeout(r, 200));
    }
    if (!ready) throw Error("Audit initialization failed");
    await call("Emulation.setDeviceMetricsOverride", {
      width: 1440,
      height: 1250,
      deviceScaleFactor: 1,
      mobile: false,
    });
    const result =
      await evaluate(`(async()=>{const gold=${JSON.stringify(gold)};let mathematicalChecks=0,domChecks=0;let stateMap=new Map(states.map(s=>[[s.cohort,s.episode_id,s.step,s.bank].join('|'),s]));for(const row of gold){let s=stateMap.get(row.slice(0,4).join('|')),r=evaluate(s,row[4],row[5]);if(r.chosen!==row[6])throw Error('Choice differs');for(const [i,k] of ['success','base','gain','help','harm','regret','opportunity'].entries()){if(Math.abs(r[k]-row[7+i])>1e-12)throw Error('Browser metric differs:'+k);mathematicalChecks++}}
    $('cohort').value='all_development';for(const s of states){$('bank').value=s.bank;updateStates();$('state').value=s.episode_id+'/'+s.step;for(const n of [4,8]){ $('size').value=String(n);options();for(const method of ['reference','uniform','offline_best','fixed:'+(n-1),'legacy__global__predicted','global__linear__predicted_visual']){$('selector').value=method;render();let r=evaluate(s,n,method),rows=Array.from($('candidates').children);if(rows.length!==n||DECISION_AUDIT.chosen()!==r.chosen)throw Error('DOM choice differs');for(let i=0;i<n;i++){if(Math.abs(Number(rows[i].children[2].textContent)-r.means[i])>.00051)throw Error('DOM outcome differs');let hit=s.first_hit.map(t=>t[i]===null?'—':t[i]).join(' / ');if(rows[i].children[3].textContent!==hit)throw Error('DOM timing differs');if(rows[i].children[4].textContent!==(s.forecast_supported[i]?'Yes':'No'))throw Error('DOM support differs')}domChecks++}}}
    for(const type of ['helpful','harmful','tied_failure','tied_success']){tour(type);let r=evaluate(DECISION_AUDIT.current(),8,$('selector').value);if(type==='helpful'&&!(r.opportunity>0&&$('selector').value==='offline_best'))throw Error('Helpful tour wrong');if(type==='harmful'&&!(r.gain<0))throw Error('Harmful tour wrong');if(type==='tied_failure'&&!r.means.every(v=>v===0))throw Error('Failure tour wrong');if(type==='tied_success'&&!r.means.every(v=>v===1))throw Error('Success tour wrong')}
    $('size').value='8';options();$('selector').value='fixed:7';render();$('size').value='4';render();if($('selector').value!=='reference')throw Error('Out-of-bank selector remains active');
    const images=[...new Set(states.flatMap(s=>Object.values(s.images)))];await Promise.all(images.map(async src=>{let i=new Image();i.src=src;await i.decode();let n=src.includes('_vla.png')?480:224;if(i.naturalWidth!==n||i.naturalHeight!==n)throw Error('Native dimensions changed')}));
    const videos=[...new Set(states.map(s=>s.video))];for(const src of videos){await new Promise((resolve,reject)=>{let v=document.createElement('video');v.preload='auto';v.muted=true;v.src=src;let timer=setTimeout(()=>reject(Error('Source video timeout')),15000);v.onloadeddata=()=>{clearTimeout(timer);if(v.videoWidth!==480||v.videoHeight!==480)return reject(Error('Source video dimensions changed'));v.removeAttribute('src');v.load();resolve()};v.onerror=()=>{clearTimeout(timer);reject(Error('Source video failed to decode'))}})}
    tour('harmful');let v=$('video');await new Promise((resolve,reject)=>{if(v.readyState>=2)return resolve();let timer=setTimeout(()=>reject(Error('Playback timeout')),15000);v.onloadeddata=()=>{clearTimeout(timer);resolve()}});v.muted=true;await v.play();await new Promise(r=>setTimeout(r,300));v.pause();if(v.currentTime<=0)throw Error('Playback did not advance');
    return {status:'passed',mathematicalChecks,domChecks,guideCases:4,nativeImagesDecoded:images.length,sourceVideosDecoded:videos.length,sourceVideoPlayback:true,outOfBankChoiceDisabled:true,models:42,states:54};})()`);
    const screen = await call("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: false,
    });
    fs.writeFileSync(
      path.join(root, "desktop.png"),
      Buffer.from(screen.data, "base64"),
    );
    await call("Emulation.setDeviceMetricsOverride", {
      width: 390,
      height: 844,
      deviceScaleFactor: 1,
      mobile: true,
    });
    result.mobile = await evaluate(
      "({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})",
    );
    if (result.mobile.width !== 390 || result.mobile.scrollWidth > 390)
      throw Error("Mobile overflow or expanded viewport");
    const mobile = await call("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: false,
    });
    fs.writeFileSync(
      path.join(root, "mobile.png"),
      Buffer.from(mobile.data, "base64"),
    );
    fs.writeFileSync(
      path.join(root, "browser_verification.json"),
      JSON.stringify(result, null, 2) + "\n",
    );
    console.log(JSON.stringify(result));
  } finally {
    await call("Browser.close").catch(() => {});
    socket.close();
  }
}
main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
