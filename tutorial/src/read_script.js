  // step 2: street views of the Read example (PAVE_DATA.street_views)
  const grid = document.getElementById("gsvGrid");
  if (grid && D.street_views) {
    const label = f => {
      const m = f.match(/^gsv_([A-Z]+)_(near|mid|far)_(forward|look_back)$/);
      return m ? "approach " + m[1] + " · " + m[2] + " · " + (m[3] === "forward" ? "toward the junction" : "look-back") : f;
    };
    grid.innerHTML = D.street_views.map(f => {
      const p = "images/read/source/street_views/" + f + ".jpg", t = label(f);
      return SINGLE ? '<figure><img data-img="' + p + '" alt="' + t + '"><figcaption>' + t + "</figcaption></figure>"
                    : '<figure><a href="' + p + '" target="_blank" rel="noopener"><img src="' + p + '" alt="' + t + '" loading="lazy"></a><figcaption>' + t + "</figcaption></figure>";
    }).join("");
  }

  // embedded images are decoded only when they come near the screen
  if (SINGLE) {
    const lazy = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { const im = e.target; im.src = imgSrc(im.dataset.img); lazy.unobserve(im); } }), {rootMargin: "800px"});
    document.querySelectorAll("img[data-img]").forEach(im => lazy.observe(im));
  }

  // read flow cards: each step with the number of its calls in this run
  const flow = document.getElementById("readFlow");
  if (flow) {
    const further = ["detail", "sharedask", "sharedaskdetail", "addlane", "reuse", "redetail", "keepuse", "keepdetail", "widekeepuse",
                     "widekeepusedetail", "leftlane", "leftuse", "leftdetail", "leftkeepuse", "leftkeepdetail", "wideleftkeepuse",
                     "wideleftkeepusedetail", "leftask", "leftaskdetail", "leftaskkeep", "leftaskkeepdetail"];
    const count = steps => D.calls.filter(c => c.pipeline === "read" && steps.includes(c.step)).length;
    const calls = (steps, none) => { const k = count(steps); return k ? k + (k === 1 ? " call" : " calls") : none; };
    const cards = [["r1", "1 Network node and arms", "no calls"], ["r2", "2 Imagery", "no calls"],
                   ["r3", "3 Trace", calls(["trace"], "0 calls")], ["r4", "4 Blind reading", calls(["read"], "0 calls")],
                   ["r5", "5 Re-trace", calls(["retrace"], "0 calls here")], ["r6", "6 Lane use", calls(["use"], "0 calls")],
                   ["r7", "7 Further checks", calls(further, "0 calls here")], ["r8", "8 Result", "no calls"]];
    flow.innerHTML = cards.map(s => '<a href="#' + s[0] + '">' + s[1] + "<small>" + s[2] + "</small></a>").join("");
  }

