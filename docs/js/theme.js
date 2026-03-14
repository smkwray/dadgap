/* dadgap — theme + Chart.js */
(function () {
  'use strict';
  function getSaved() { try { return localStorage.getItem('dadgap-theme'); } catch(e) { return null; } }
  function apply(t) {
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem('dadgap-theme', t); } catch(e) {}
    var b = document.getElementById('themeToggle'); if (b) b.textContent = t==='dark'?'\u2600':'\u263E';
  }
  apply(getSaved() || 'light');
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) { if (!getSaved()) apply(e.matches?'dark':'light'); });
  window.dgToggle = function() { var c=document.documentElement.getAttribute('data-theme')||'light'; apply(c==='dark'?'light':'dark'); if(window.dgRebuild) window.dgRebuild(); };
  window.dgDark = function() { return document.documentElement.getAttribute('data-theme')==='dark'; };
  window.C = function() {
    var d = dgDark();
    return { navy:d?'#c5cfe0':'#1a1a2e', accent:d?'#7ba0e0':'#537ec5', red:d?'#ef5350':'#c0392b', green:d?'#66bb6a':'#27ae60', orange:d?'#ffa726':'#e67e22', purple:d?'#ab47bc':'#8e44ad', teal:d?'#26a69a':'#16a085', dark:d?'#90a4ae':'#2c3e50', muted:d?'#3a4050':'#d1d5db', text:d?'#d4d8e0':'#1f2937', textSec:d?'#9ca3af':'#6b7280', grid:d?'rgba(255,255,255,0.08)':'rgba(0,0,0,0.08)', bg:d?'#1a1e2c':'#ffffff', palette:d?['#7ba0e0','#ffa726','#66bb6a','#ef5350','#ab47bc','#90a4ae','#26a69a','#ff7043']:['#537ec5','#e67e22','#27ae60','#c0392b','#8e44ad','#2c3e50','#16a085','#d35400'] };
  };
  window.dgDefaults = function() {
    if(typeof Chart==='undefined') return;
    try { var c=C(); Chart.defaults.font.family="-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif"; Chart.defaults.font.size=13; Chart.defaults.color=c.textSec; Chart.defaults.borderColor=c.grid;
      if(Chart.defaults.plugins&&Chart.defaults.plugins.legend&&Chart.defaults.plugins.legend.labels) Chart.defaults.plugins.legend.labels.color=c.textSec;
      if(Chart.defaults.plugins&&Chart.defaults.plugins.tooltip){Chart.defaults.plugins.tooltip.backgroundColor=c.navy;Chart.defaults.plugins.tooltip.titleColor='#fff';Chart.defaults.plugins.tooltip.bodyColor='#e0e0e0';}
    } catch(e){}
  };
})();
