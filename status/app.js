const $ = id => document.getElementById(id);
const sections = document.querySelectorAll('.section');
document.querySelectorAll('.nav').forEach(btn => btn.addEventListener('click', () => goTo(btn.dataset.section)));

function goTo(id){
  sections.forEach(s => s.classList.toggle('active', s.id === id));
  document.querySelectorAll('.nav').forEach(n => n.classList.toggle('active', n.dataset.section === id));
  window.scrollTo({top:0, behavior:'smooth'});
}

function setProduct(id){ $('productId').value=id; goTo('prediction'); }

async function loadDashboard(){
  try{
    const h = await fetch('/api/health').then(r=>r.json());
    if(!h.online){
      $('systemText').textContent='MODEL FILES MISSING';
      $('message').textContent='Run: python prepare_model.py --data retail_store_inventory.csv';
      return;
    }
    const s = await fetch('/api/stats').then(r=>r.json());
    $('rows').textContent = Number(s.rows).toLocaleString();
    $('products').textContent = Number(s.products).toLocaleString();

    const p = await fetch('/api/products').then(r=>r.json());
    const list = $('productList');
    (p.products || []).forEach(id => {
      const o = document.createElement('option'); o.value=id; list.appendChild(o);
    });
  }catch(e){
    $('systemText').textContent='API OFFLINE';
  }
}

async function predict(){
  const product_id = $('productId').value.trim();
  if(!product_id){ $('message').textContent='Enter a Product ID first.'; return; }

  const btn = $('predictBtn');
  btn.disabled=true; btn.textContent='ANALYZING...';
  $('message').textContent='AI model is processing the product...';
  $('resultStatus').textContent='RUNNING';

  try{
    const response = await fetch('/predict',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({product_id})
    });
    const data = await response.json();
    if(!response.ok) throw new Error(data.error || 'Prediction failed');

    $('predicted').textContent=data.predicted_demand.toFixed(2);
    $('actual').textContent=data.actual_demand.toFixed(2);
    $('inventory').textContent=data.inventory.toFixed(2);
    $('sold').textContent=data.units_sold.toFixed(2);
    $('price').textContent=data.price.toFixed(2);
    $('resultStatus').textContent=data.status;
    $('recommendation').textContent='◆ '+data.recommendation;
    $('message').textContent='Prediction completed for '+data.product_id+'.';
  }catch(err){
    $('resultStatus').textContent='ERROR';
    $('message').textContent=err.message;
  }finally{
    btn.disabled=false; btn.textContent='RUN PREDICTION';
  }
}
$('predictBtn').addEventListener('click', predict);
$('productId').addEventListener('keydown', e => {if(e.key==='Enter') predict();});
loadDashboard();
