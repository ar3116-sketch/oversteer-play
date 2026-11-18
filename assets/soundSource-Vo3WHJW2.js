const t=`// The sound of a car, made from what a car does. Pure arithmetic on arrays of samples: no Web Audio,
// no DOM, no imports. This exact file runs in two places:
//   - in the game, inside an AudioWorklet (audio.js loads this file's text into the audio thread)
//   - in node, where tools/sound-render.mjs writes WAV files with it and measures them
// so what is measured is what is heard.
//
// THE ENGINE (EngineVoice)
// A crank turns. Each cylinder fires at its own crank angle, set by the layout's firing order. When
// its exhaust valve opens, the burnt gas blows down into the manifold: one pressure pulse. The pulse
// reaches the collector after travelling its own length of pipe, so unequal pipes give unevenly spaced
// pulses. From there the pulses run through the header and the tailpipe (two tubes that ring), a
// silencer (which the Exhaust upgrade removes), and out of the tailpipe, which turns flow into sound.
// Nothing here is an oscillator at "engine pitch": the note is simply how often the pulses come.
//   revs            how often pulses come
//   load            how big and how sharp each pulse is. Off the throttle the cylinders only pump
//                   air, so the pulses go small and soft and the pipes sound hollow.
//   every pulse     differs a little from the last, more so at idle (that is what makes it alive)
// The same firing events also knock on the engine block, tick the valve gear and pull gulps of air
// through the intake, which roars as the throttle opens. A turbo spools with exhaust flow, whistles,
// and chuffs or flutters when the throttle shuts. Unburnt fuel pops in an open exhaust on the overrun.
//
// THE REST OF THE CAR (CarFx): tyres on each kind of ground, wind, landings, knocks.
// THE ELECTRIC CAR (EvVoice): motor and inverter tones that follow motor speed and torque.
// THE MIXER (Mixer): every voice added together, kept from clipping. It is driven by small messages,
// the same ones the game posts to the audio thread, so node can play a session exactly as the game does.

const TAU = Math.PI * 2;
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, t) => a + (b - a) * t;
const frac = (v) => v - Math.floor(v);

// How often the slow things (smoothing, filter settings, decisions) are worked out: every CTL samples.
export const CTL = 32;
const DN = 1e-20;                      // keeps silent filters away from denormal numbers, which are slow

// sin(2 pi x) from a table, x in 0..1
const SINE_N = 2048;
const SINE = new Float32Array(SINE_N + 1);
for (let i = 0; i <= SINE_N; i++) SINE[i] = Math.sin((i / SINE_N) * TAU);

const onePole = (hz, sr) => 1 - Math.exp((-TAU * Math.min(hz, sr * 0.45)) / sr);   // y += k (x - y)
const decayK = (seconds, sr) => Math.exp(-1 / (Math.max(seconds, 1e-5) * sr));      // x *= k dies away in \`seconds\`
const svfF = (hz, sr) => 2 * Math.sin((Math.PI * Math.min(hz, sr * 0.16)) / sr);     // tuning of the small band-pass filters

class Rng {
  constructor(seed) { this.s = (seed | 0) || 0x1234567; }
  next() { let s = this.s; s ^= s << 13; s ^= s >>> 17; s ^= s << 5; this.s = s; return (s >>> 0) / 4294967296; }
  tri() { return this.next() + this.next() - 1; }           // -1..1, mostly near 0
}

// ================================================================================================
// The engine
// ================================================================================================
const IGN = 0, EXH = 1, INT = 2;
const K_IGN0 = 0, K_IGN1 = 1, K_INT0 = 2, K_INT1 = 3, K_TICK = 4, K_POP = 5;

// How much of each part of the engine is heard from each camera: [exhaust, cabin muffling, intake,
// mechanical, turbo, gear whine]. Chase, far chase, bonnet, cockpit.
const VIEW = [
  [1.0, 0.0, 0.55, 0.3, 0.8, 0.45],
  [0.85, 0.0, 0.4, 0.2, 0.6, 0.3],
  [0.72, 0.3, 0.85, 0.6, 1.0, 0.7],
  [0.55, 0.85, 0.8, 0.6, 1.0, 1.0],
];

export class EngineVoice {
  // spec: from engineSpecs.js. opts.lite: a cheaper voice for other people's cars (exhaust only).
  constructor(spec, sr, opts = {}) {
    const sp = this.spec = spec;
    this.sr = sr; this.lite = !!opts.lite;
    this.rng = new Rng((sp.seed ^ (opts.seed || 0x5bd1e995)) | 0);
    this.dtc = CTL / sr; this.ctl = 0;
    this.tap = null;                 // tools may set this to five arrays to hear the parts apart

    // ---- the cycle: every cylinder's spark, exhaust opening and intake opening, in crank order
    const n = sp.cyl.length;
    this.cylGain = new Float32Array(n); this.cylDelay = new Float32Array(n); this.cylLope = new Float32Array(n);
    this.cylSmear = new Uint16Array(n); this.missed = new Uint8Array(n);
    const evs = [];
    let far = 0;
    for (let c = 0; c < n; c++) {
      const cy = sp.cyl[c];
      this.cylGain[c] = cy.gain; this.cylLope[c] = cy.lope || 0;
      this.cylDelay[c] = 1.5 + cy.delay * sr;                       // samples from valve to collector
      this.cylSmear[c] = Math.max(1, Math.round((cy.smear || 0) * sr));
      far = Math.max(far, this.cylDelay[c] + this.cylSmear[c]);
      evs.push([frac(cy.phase + sp.evo), EXH, c]);
      if (!this.lite) { evs.push([frac(cy.phase), IGN, c]); evs.push([frac(cy.phase + sp.ivo), INT, c]); }
    }
    evs.sort((a, b) => a[0] - b[0]);
    this.nEv = evs.length;
    this.evPhase = new Float64Array(this.nEv); this.evType = new Uint8Array(this.nEv); this.evCyl = new Uint8Array(this.nEv);
    evs.forEach((e, i) => { this.evPhase[i] = e[0]; this.evType[i] = e[1]; this.evCyl[i] = e[2]; });
    this.ph = this.rng.next(); this.dph = 0;
    this.ev = 0; while (this.ev < this.nEv && this.evPhase[this.ev] < this.ph) this.ev++;

    // pulses are written into this ring at the moment they will reach the collector
    let size = 64; while (size < far + 0.004 * sr + 64) size <<= 1;
    this.ring = new Float32Array(size); this.mask = size - 1; this.w = 0;
    this.kick = new Float32Array(6);

    // ---- exhaust
    this.e = 0; this.b = 0; this.m = 0; this.kd = 0.9; this.ka = 0.5; this.kb = 0.99;
    this.seed = (sp.seed * 2654435761) | 0 || 1; this.hpN = 0; this.kN = onePole(1500, sr); this.rz = 0; this.kRz = onePole(3800, sr);
    this.pe = 0; this.kPop = decayK(0.003, sr);
    this.hLen = Math.max(2, Math.round(sp.header.time * sr)); this.hBuf = new Float32Array(this.hLen); this.hi = 0; this.hlp = 0;
    this.kH = onePole(sp.header.lossHz, sr); this.rH = sp.header.refl;
    this.pLen = Math.max(2, Math.round(sp.pipe.time * sr)); this.pBuf = new Float32Array(this.pLen); this.pi = 0; this.plp = 0;
    this.kP = onePole(sp.pipe.lossHz, sr); this.rP = sp.pipe.refl;
    this.m1 = 0; this.m2 = 0; this.kM = onePole(sp.muffler.hz, sr);
    this.rad = 0; this.kR = onePole(sp.radHz, sr);
    this.c1 = 0; this.c2 = 0; this.kC = onePole(650, sr);

    // ---- intake: two resonances (the airbox or plenum, and the runners) rung by each gulp of air
    // (each resonance is fed the change in its input, so it rings at its own note and passes nothing steady)
    const reso = (hz, q) => { const th = (TAU * hz) / sr, r = Math.exp((-Math.PI * (hz / q)) / sr); return [2 * r * Math.cos(th), r * r]; };
    const I = sp.intake;
    [this.a11, this.a12] = reso(I.f1, I.q1);
    [this.a21, this.a22] = reso(I.f2, I.q2);
    this.g1 = 0.5; this.g2 = 0.5 * I.mix2;
    this.is = 0; this.isa = 0; this.isb = 0; this.kIs = onePole(650, sr); this.i1a = 0; this.i1b = 0; this.i2a = 0; this.i2b = 0;
    this.hl = 0; this.hb = 0; this.hf = 0.2; this.hissG = 0;

    // ---- block and valve gear
    const M = sp.mech;
    [this.b11, this.b12] = reso(M.f1, 2.2);
    [this.b21, this.b22] = reso(M.f2, 3);
    this.gK1 = 1; this.gK2 = 0.8;
    this.ks = 0; this.kKs = onePole(1500, sr); this.xga = 0; this.xgb = 0; this.k1a = 0; this.k1b = 0; this.k2a = 0; this.k2b = 0;
    this.te = 0; this.kTick = decayK(0.0012, sr); this.wl = 0; this.kWh = onePole(1400, sr); this.whirrG = 0;
    this.fph = 0; this.fInc = 0; this.fanG = 0;

    // ---- turbo, blower, gears
    this.spool = 0; this.tl = 0; this.tb = 0; this.tf = 0.3; this.wph = 0; this.wInc = 0; this.gWhN = 0; this.gWhT = 0;
    this.bl = 0; this.bb = 0; this.bf = 0.2; this.bovS = 0; this.bovTo = 0; this.bovEnv = 0; this.bovKind = 0; this.bovPh = 0; this.bovRate = 0; this.armed = false;
    this.sph = 0; this.sInc = 0; this.scG = 0;
    this.q1 = 0; this.q1Inc = 0; this.q2 = 0; this.q2Inc = 0; this.wA1 = 0; this.wA2 = 0;

    // ---- the driver's side of it
    // (boost: the physics' own turbo boost 0..1 when the game sends it; -1 = none sent, the voice spools its own)
    this.t = { rpm: sp.idle, thr: 0, gear: 0, shiftT: 0, cam: 0, speed: 0, boost: -1 };
    this.rpm = sp.idle; this.open = 0; this.load = 0; this.gear = 0; this.shiftT = 0; this.shiftDir = 0; this.blipT = 0;
    this.lim = false; this.limPh = 0; this.dip = 0; this.wasCut = false; this.over = false; this.overT = 0;
    this.wob = 0; this.wobTo = 0; this.wobT = 0; this.revHz = 0;
    this.ampFire = 0; this.ampInt = 0; this.ampIgn = 0; this.tickAmp = 0; this.sigma = 0.1; this.pFire = 1; this.lopeNow = 0; this.cycleK = 1;
    this.jitter = 0;
    this.kRpm = 1 - Math.exp(-this.dtc / 0.02); this.kUp = 1 - Math.exp(-this.dtc / 0.02); this.kDn = 1 - Math.exp(-this.dtc / 0.045);
    this.kDip = 1 - Math.exp(-this.dtc / 0.012); this.kWob = 1 - Math.exp(-this.dtc / 0.15); this.kView = 1 - Math.exp(-this.dtc / 0.09);
    this.mixTo = new Float32Array(6); this.mix = new Float32Array(6);
    this._view(0); this.mix.set(this.mixTo);
  }

  // What the game knows this frame. Missing fields keep their last value.
  setState(st) {
    const t = this.t;
    if (st.rpm !== undefined) t.rpm = clamp(+st.rpm || 0, 0, 1.05);
    if (st.thr !== undefined) t.thr = clamp(+st.thr || 0, 0, 1);
    if (st.gear !== undefined) t.gear = st.gear | 0;
    if (st.shiftT !== undefined) this.shiftT = Math.max(0, +st.shiftT || 0);
    if (st.speed !== undefined) t.speed = Math.abs(+st.speed || 0);
    if (st.boost !== undefined) t.boost = +st.boost >= 0 ? clamp(+st.boost, 0, 1.2) : -1;
    if (st.cam !== undefined && (st.cam | 0) !== t.cam) { t.cam = clamp(st.cam | 0, 0, 3); this._view(t.cam); }
  }

  // Jump straight to a steady state (used to measure loudness, and when a car appears mid-drive).
  prime(st) {
    this.setState(st);
    this.rpm = this.t.rpm; this.open = this.t.thr; this.gear = this.t.gear;
    if (this.spec.turbo) this.spool = this.t.boost >= 0 ? this.t.boost : clamp(this.t.thr * (0.12 + 1.3 * this.rpm), 0, 1);
    this.mix.set(this.mixTo);
    this.ctl = 0;
  }

  _view(cam) {
    const sp = this.spec, v = VIEW[cam] || VIEW[0], to = this.mixTo, pos = sp.view.enginePos;
    for (let i = 0; i < 6; i++) to[i] = v[i];
    if (pos > 0) {
      // engine behind the driver: the chase camera hears the engine itself, the bonnet camera does not
      if (cam < 2) { to[2] *= pos === 2 ? 1.5 : 1.3; to[3] *= pos === 2 ? 1.5 : 1.3; }
      else if (cam === 2) { to[2] *= 0.6; to[3] *= 0.5; }
      else { to[2] *= pos === 1 ? 1.2 : 0.9; to[3] *= pos === 1 ? 1.25 : 0.9; }
    }
    if (sp.view.openTop && cam === 3) { to[0] = 0.8; to[1] = 0.2; }
  }

  // ---- the slow half: decide what the engine is doing for the next CTL samples
  _control() {
    const sp = this.spec, sr = this.sr, dt = this.dtc, t = this.t, rng = this.rng;
    this.rpm += (t.rpm - this.rpm) * this.kRpm;
    const rpm = this.rpm, idle = sp.idle;

    // gearbox: which way the lever went
    if (t.gear !== this.gear) {
      if (this.gear > 0 && t.gear > 0) {
        this.shiftDir = t.gear > this.gear ? 1 : -1;
        if (this.shiftDir < 0) this.blipT = Math.max(0.17, this.shiftT);       // on the way down the driver blips the throttle to match revs
      } else this.shiftDir = 0;
      this.gear = t.gear;
    }
    const shifting = this.shiftT > 0;
    if (shifting) this.shiftT -= dt;
    let thr = t.thr, cut = false;
    if (shifting && this.shiftDir > 0) {
      if (sp.flatShift) cut = true;            // competition gearbox: foot stays down, the spark is cut for an instant
      else thr = 0;                            // road gearbox: lift, change, back on
    }
    if (this.blipT > 0) { this.blipT -= dt; const bl = sp.blip * Math.min(1, this.blipT / 0.06); if (bl > thr) thr = bl; }

    // rev limiter: the spark is cut in bursts, the revs sag and come back ("bap-bap-bap")
    if (this.lim) { if (t.rpm < 0.962 || thr < 0.25) this.lim = false; }
    else if (t.rpm >= 0.985 && thr > 0.4) { this.lim = true; this.limPh = 0; }
    if (this.lim) {
      this.limPh += dt * sp.limiter.hz; if (this.limPh >= 1) this.limPh -= 1;
      if (this.limPh < sp.limiter.duty) cut = true;
    }
    this.dip += ((this.lim && cut ? sp.limiter.dip : 0) - this.dip) * this.kDip;

    // throttle plate and what it lets into the cylinders
    this.open += (thr - this.open) * (thr > this.open ? this.kUp : this.kDn);
    const nearIdle = clamp(1 - (rpm - idle) / 0.07, 0, 1);
    const load = Math.max(this.open, sp.idleLoad * nearIdle);       // the idle valve keeps it turning over
    this.load = load;
    const charge = Math.pow(load, 0.6);

    // turbo: spun by exhaust flow, so it follows load and revs, late
    const tb = sp.turbo;
    let boost = 1;
    if (tb) {
      if (t.boost >= 0) {
        // the physics' boost (game/stats.js stepBoost, sent with each frame): the whistle, the push in the exhaust and
        // the blow-off follow the car's real surge and lull. Smoothed only over the gap between frames.
        this.spool += (t.boost - this.spool) * Math.min(1, dt / 0.03);
      } else {
        // nothing sent (the garage's showroom, a tool): the voice's own guess, from load and revs
        const want = cut ? this.spool : clamp(load * (0.12 + 1.3 * rpm), 0, 1);
        this.spool += (want - this.spool) * Math.min(1, dt / (want > this.spool ? tb.lag : tb.fall));
        if (tb.antilag && load < 0.1 && rpm > idle + 0.12 && this.spool < 0.55) this.spool += (0.55 - this.spool) * Math.min(1, dt / 0.15);
      }
      boost = (1 + tb.boost * this.spool) / (1 + tb.boost);
    }

    const over = !cut && load < 0.07 && rpm > idle + 0.09;           // coasting down in gear, throttle shut
    if (over) this.overT += dt; else this.overT = 0;
    this.over = over;

    // how hard each cylinder fires, and how much one firing differs from the next
    this.ampFire = cut ? sp.pump * 1.4 : sp.pump + (1 - sp.pump) * charge * boost;
    this.sigma = lerp(sp.sigmaIdle, sp.sigmaLoad, Math.min(1, load * 1.3));
    this.pFire = sp.misfire > 0 ? 1 - sp.misfire * clamp(1 - load * 1.2, 0, 1) : 1;
    this.lopeNow = sp.lope * clamp(1 - (rpm - idle) / 0.16, 0, 1) * (1 - 0.7 * this.open);
    this.jitter = (0.00002 + 0.00025 * this.sigma) * sr;

    // unburnt fuel lighting in the pipe: crackle on the overrun, bangs from anti-lag, a pop as the spark returns
    let popRate = 0;
    if (over && sp.crackle > 0) popRate = sp.crackle * (4 + 36 * rpm) * (Math.exp(-this.overT / 0.9) + 0.08);
    if (tb && tb.antilag && load < 0.1 && rpm > idle + 0.12) popRate += 10 + 14 * rpm;
    if (popRate > 0 && rng.next() < popRate * dt) { const r = rng.next(); this._pop((0.2 + 1.3 * r * r) * sp.popGain); }
    if (this.wasCut && !cut && sp.crackle > 0.2 && rng.next() < 0.3 + 0.6 * sp.crackle) this._pop((0.3 + 0.5 * rng.next()) * sp.popGain);
    this.wasCut = cut;

    // crank speed, with the idle hunting a little (a lumpy cam makes it surge)
    if ((this.wobT -= dt) <= 0) { this.wobTo = rng.tri(); this.wobT = 0.1 + 0.3 * rng.next(); }
    this.wob += (this.wobTo - this.wob) * this.kWob;
    let rpmEff = rpm * (1 - this.dip) * (1 + this.wob * sp.hunt * nearIdle);
    if (rpmEff < idle * 0.85) rpmEff = idle * 0.85;
    const revHz = (rpmEff * sp.redline) / 60;
    this.revHz = revHz;
    this.dph = revHz / (sp.stroke === 2 ? 1 : 2) / sr;

    // the shape of one exhaust pulse. Valve events take a fixed number of crank degrees, so pulses get
    // shorter as revs rise; at low revs the cylinder simply empties, which takes a fixed time.
    const P = sp.pulse, degT = 1 / (revHz * 360);
    const tcD = Math.min(P.decayMax, P.decayDeg * degT);
    const tcA = clamp(P.attackDeg * degT * (1.9 - 1.1 * charge) * (cut ? 1.5 : 1), P.attackMin, tcD * 0.8);
    this.kd = Math.exp(-1 / (tcD * sr)); this.ka = 1 - Math.exp(-1 / (tcA * sr));
    this.kb = Math.exp(-1 / (Math.min(0.02, P.bodyDeg * degT) * sr));

    if (!this.lite) {
      const I = sp.intake, M = sp.mech;
      this.ampInt = -0.4 * I.level * Math.pow(this.open, 1.3) * (0.35 + 0.65 * rpm) * (0.7 + 0.3 * boost);
      this.hf = svfF(900 + 1900 * rpm, sr);
      this.hissG = I.hiss * this.open * (0.12 + 0.88 * rpm) * 0.22;
      this.ampIgn = (cut ? 0.1 : 0.2 + 0.8 * charge) * (0.45 + 0.55 * rpm) * M.level;
      this.tickAmp = M.tick * (0.3 + 0.7 * rpm) * 0.08;
      this.whirrG = M.whirr * Math.pow(rpm, 1.5) * 0.06;
      this.fInc = (M.fan.order * revHz) / sr; this.fanG = M.fan.level * (0.25 + 0.75 * rpm) * 0.12;
      if (this.fInc > 0.4) this.fanG = 0;

      if (tb) {
        // whistle: mostly a resonance blown by noise (breath across a bottle), with a thread of pure tone
        const s = this.spool, hz = tb.f0 + (tb.f1 - tb.f0) * Math.pow(s, 0.85);
        this.tf = svfF(hz, sr); this.wInc = Math.min(hz, sr * 0.45) / sr;
        const lv = tb.level * s * s * (0.5 + 0.5 * load);
        this.gWhN = lv * 0.017; this.gWhT = lv * 0.003;
        // throttle shut on a spinning turbo: the air has to go somewhere
        // (armed again only with the foot back down: the plate takes a few ms to close, and re-arming on it alone
        // fired the valve three or four times per lift, each one cutting the spool away from the physics' boost)
        if (this.armed && thr < 0.2) { this.armed = false; if (s > 0.3 && tb.bov) this._bov(s); }
        else if (thr >= 0.2 && this.open > 0.6) this.armed = true;
        if (this.bovEnv > 1e-4) {
          if (this.bovKind === 2) {
            // no blow-off valve: the air backs out through the compressor in chops, slowing as it goes
            this.bovPh += this.bovRate * dt; this.bovRate += (8 - this.bovRate) * Math.min(1, dt / 0.25);
            if (this.bovPh >= 1) this.bovPh -= 1;
            this.bovTo = this.bovEnv * Math.exp(-this.bovPh * 5.5);
            this.bovEnv *= Math.exp(-dt / 0.3);
            this.bf = svfF(1350 + 500 * this.bovEnv, sr);
          } else {
            // a valve dumps it in one breath, falling in pitch
            this.bovTo = this.bovEnv;
            this.bovEnv *= Math.exp(-dt / 0.14);
            this.bf = svfF(1300 + 2400 * Math.min(1, this.bovEnv * 1.5), sr);
          }
        } else { this.bovEnv = 0; this.bovTo = 0; }
      }
      if (sp.sc) { this.sInc = (sp.sc.order * revHz) / sr; this.scG = this.sInc < 0.4 ? sp.sc.level * (0.25 + 0.75 * load) * (0.3 + 0.7 * rpm) * 0.05 : 0; }
      const W = sp.whine;
      if (W.level > 0) {
        // straight-cut gears: the pair in use sings at revs x teeth, the final drive follows road speed
        const g = this.gear > 0 ? this.gear : 1;
        const f1 = revHz * W.teeth[Math.min(g, W.teeth.length) - 1], f2 = t.speed * W.finalHz;
        const torque = 0.35 + 0.65 * Math.max(charge, over ? 0.5 : 0);
        this.q1Inc = f1 / sr; this.q2Inc = f2 / sr;
        this.wA1 = f1 < sr * 0.2 ? W.level * torque * 0.03 * clamp(f1 / 300, 0, 1) : 0;
        this.wA2 = f2 < sr * 0.2 ? W.level * torque * 0.045 * clamp(f2 / 200, 0, 1) : 0;
      } else { this.wA1 = 0; this.wA2 = 0; }
    }
    const mx = this.mix, to = this.mixTo, kv = this.kView;
    for (let i = 0; i < 6; i++) mx[i] += (to[i] - mx[i]) * kv;
    if (this.te < 1e-12) this.te = 0;
    if (this.pe < 1e-12) this.pe = 0;
  }

  _bov(s) {
    const tb = this.spec.turbo;
    // (0.36: a lift on full boost is as loud as it was when the valve fired several times, the last at about a third
    // of the spool; now it fires once, so a lift on half boost is quieter than one on full)
    this.bovKind = tb.bov; this.bovEnv = clamp(0.36 * s * tb.bovLevel, 0, 1.6); this.bovPh = 0; this.bovRate = 24 + 6 * this.rng.next();
    this.spool *= tb.bov === 2 ? 0.45 : 0.7;
  }

  // Unburnt fuel going off in the pipe. Unlike a cylinder emptying through a valve it has no gentle
  // start, so it goes straight in as flow: that hard edge is the crack.
  _pop(a) { this.kick[K_POP] += a; }

  // one valve or spark event, \`late\` samples ago (a fraction: events fall between samples)
  _fire(ev, late, w) {
    const type = this.evType[ev], c = this.evCyl[ev], rng = this.rng, kick = this.kick;
    if (type === EXH) {
      let a = this.ampFire * this.cylGain[c] * this.cycleK * (1 + this.sigma * rng.tri());
      if (this.lopeNow > 0) a *= Math.max(0.1, 1 + this.lopeNow * (0.6 * this.cylLope[c] + 0.5 * rng.tri()));
      if (this.over) { const r = rng.next(); a *= 0.4 + 2.2 * r * r; }           // the odd cylinder still catches: burble
      else if (this.pFire < 1) {
        if (rng.next() > this.pFire) { a = this.spec.pump * this.cylGain[c] * 0.7; if (this.missed[c] < 4) this.missed[c]++; }
        else { a *= 1 + 0.22 * this.missed[c]; this.missed[c] = 0; }               // after a miss the next one has more to burn
      }
      if (a < 0) a = 0;
      const d = this.cylDelay[c] + this.jitter * rng.next() - late, k = d | 0, f = d - k;
      const ring = this.ring, mask = this.mask, sm = this.cylSmear[c];
      if (sm <= 1) { ring[(w + k) & mask] += a * (1 - f); ring[(w + k + 1) & mask] += a * f; }
      else {
        const s = a / sm;
        ring[(w + k) & mask] += s * (1 - f);
        for (let j = 1; j < sm; j++) ring[(w + k + j) & mask] += s;
        ring[(w + k + sm) & mask] += s * f;
      }
      if (!this.lite) kick[K_TICK] += this.tickAmp * (0.5 + rng.next());
    } else if (type === IGN) {
      const a = this.ampIgn * (0.8 + 0.4 * rng.next()) * this.cylGain[c];
      kick[K_IGN0] += a * late; kick[K_IGN1] += a * (1 - late);
    } else {
      const a = this.ampInt * (1 + 0.12 * rng.tri());
      kick[K_INT0] += a * late; kick[K_INT1] += a * (1 - late);
      kick[K_TICK] += this.tickAmp * 0.6 * (0.5 + rng.next());
    }
  }

  // Write n samples into out[0..n).
  render(out, n) {
    let i = 0;
    while (i < n) {
      if (this.ctl <= 0) { this._control(); this.ctl = CTL; }
      const m = Math.min(n - i, this.ctl);
      this._run(out, i, m);
      i += m; this.ctl -= m;
    }
  }

  _run(out, i0, m) {
    const sp = this.spec, lite = this.lite, kick = this.kick, tap = this.tap;
    let ph = this.ph, ev = this.ev; const dph = this.dph, nEv = this.nEv, evPhase = this.evPhase;
    let nextPh = ev < nEv ? evPhase[ev] : 1;
    const ring = this.ring, mask = this.mask; let w = this.w;
    let e = this.e, b = this.b, mm = this.m; const kd = this.kd, ka = this.ka, kb = this.kb, body = sp.pulse.body;
    let seed = this.seed, hpN = this.hpN, pe = this.pe, rz = this.rz; const kRz = this.kRz, kN = this.kN, rasp = sp.rasp, kPop = this.kPop;
    const hBuf = this.hBuf, hLen = this.hLen, kH = this.kH, rH = this.rH; let hi = this.hi, hlp = this.hlp;
    const pBuf = this.pBuf, pLen = this.pLen, kP = this.kP, rP = this.rP; let pi = this.pi, plp = this.plp;
    let m1 = this.m1, m2 = this.m2, rad = this.rad, c1 = this.c1, c2 = this.c2;
    const kM = this.kM, leak = sp.muffler.leak, drive = sp.drive, kR = this.kR, kC = this.kC;
    const mx = this.mix, gExh = mx[0], cab = mx[1], gInt = mx[2], gMech = mx[3], gTurbo = mx[4], gWhine = mx[5];
    const popDirect = 0.25 * sp.open;
    // intake
    let is = this.is, isa = this.isa, isb = this.isb, ks = this.ks, xga = this.xga, xgb = this.xgb, i1a = this.i1a, i1b = this.i1b, i2a = this.i2a, i2b = this.i2b, hl = this.hl, hb = this.hb;
    const kKs = this.kKs, kIs = this.kIs, a11 = this.a11, a12 = this.a12, a21 = this.a21, a22 = this.a22, g1 = this.g1, g2 = this.g2, hf = this.hf, hissG = this.hissG;
    // block
    let k1a = this.k1a, k1b = this.k1b, k2a = this.k2a, k2b = this.k2b, te = this.te, wl = this.wl, fph = this.fph;
    const b11 = this.b11, b12 = this.b12, b21 = this.b21, b22 = this.b22, gK1 = this.gK1, gK2 = this.gK2, kTick = this.kTick, kWh = this.kWh, whirrG = this.whirrG, fInc = this.fInc, fanG = this.fanG;
    // turbo, blower, gears
    const hasTurbo = !lite && !!sp.turbo, hasSc = !lite && !!sp.sc && this.scG > 0, hasWhine = !lite && (this.wA1 > 0 || this.wA2 > 0);
    let tl = this.tl, tbd = this.tb, wph = this.wph, bl = this.bl, bb = this.bb, bovS = this.bovS, sph = this.sph, q1 = this.q1, q2 = this.q2;
    const tf = this.tf, wInc = this.wInc, gWhN = this.gWhN, gWhT = this.gWhT, bf = this.bf, bovTo = this.bovTo, sInc = this.sInc, scG = this.scG;
    const scH2 = hasSc && sp.sc.h2 > 0 && sInc < 0.2 ? sp.sc.h2 : 0;
    const q1Inc = this.q1Inc, q2Inc = this.q2Inc, wA1 = this.wA1, wA2 = this.wA2;
    const bovOn = hasTurbo && (bovTo > 1e-5 || bovS > 1e-5);

    for (let i = i0, end = i0 + m; i < end; i++) {
      // ---- the crank turns; anything due since the last sample happens now
      ph += dph;
      while (ph >= nextPh) {
        if (ev < nEv) { this._fire(ev, (ph - nextPh) / dph, w); ev++; nextPh = ev < nEv ? evPhase[ev] : 1; }
        else { ph -= 1; ev = 0; nextPh = nEv > 0 ? evPhase[0] : 2; this.cycleK = 1 + 0.5 * this.sigma * this.rng.tri(); }
      }

      // ---- exhaust: pulses arriving at the collector become flow
      const x = ring[w]; ring[w] = 0; w = (w + 1) & mask;
      e = e * kd + x;                              // blow-down: the cylinder emptying
      b += ka * (e - b);                           // ...through a valve that takes a moment to open
      mm = mm * kb + x;                            // the piston pushing out the rest
      seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
      const nz = seed * 4.656612873077393e-10;
      hpN += kN * (nz - hpN);
      const nzh = nz - hpN;                        // hiss: the top of white noise
      rz += kRz * (nzh - rz);                      // ...and a band of it, 1.5 to 4 kHz, where exhaust rasp lives
      const pk = kick[K_POP];
      if (pk > 0) { kick[K_POP] = 0; e += pk; b += 0.4 * pk; }
      pe = pe * kPop + pk;
      // gas rushing past the valve is noisy, so the hiss rides on the pulses: this is the rasp
      const q = b + body * mm + rz * (b * rasp + pe * 0.6) * 1.6;
      // header, then tailpipe: each a tube that sends part of every pulse back, upside down
      hlp += kH * (hBuf[hi] - hlp);
      const y1 = q - rH * hlp + DN;
      hBuf[hi] = y1; if (++hi === hLen) hi = 0;
      plp += kP * (pBuf[pi] - plp);
      const y2 = y1 - rP * plp;
      pBuf[pi] = y2; if (++pi === pLen) pi = 0;
      // silencer
      m1 += kM * (y2 - m1); m2 += kM * (m1 - m2);
      let z = m2 + leak * (y2 - m2) + rz * pe * popDirect * 1.6;
      // loud pulses steepen as they travel: squash the peaks, more on the pressure side
      z = z > 0 ? z / (1 + drive * z) : z / (1 - 0.4 * drive * z);
      // the open end radiates changes in flow, not flow
      rad += kR * (z - rad);
      let ex = z - rad;
      // from inside the car the exhaust is a boom through the floor
      c1 += kC * (ex - c1); c2 += kC * (c1 - c2);
      ex += cab * (c2 * 1.6 - ex);
      let s = ex * gExh;

      if (!lite) {
        // ---- intake
        const xi = kick[K_INT0]; kick[K_INT0] = kick[K_INT1]; kick[K_INT1] = 0;
        is += kIs * (xi - is);
        const di = is - isb + DN; isb = isa; isa = is;
        const i1 = di + a11 * i1a - a12 * i1b; i1b = i1a; i1a = i1;
        const i2 = di + a21 * i2a - a22 * i2b; i2b = i2a; i2a = i2;
        hl += hf * hb; const hh = nz - hl - 1.4 * hb; hb += hf * hh;         // air rushing in
        const intake = i1 * g1 + i2 * g2 + hb * hissG;
        // ---- block: each firing knocks on the engine; valves tick; everything whirrs
        const xg = kick[K_IGN0]; kick[K_IGN0] = kick[K_IGN1]; kick[K_IGN1] = 0;
        ks += kKs * (xg - ks);                        // a knock is a thud, not a click
        const dg = ks - xgb + DN; xgb = xga; xga = ks;
        const k1 = dg + b11 * k1a - b12 * k1b; k1b = k1a; k1a = k1;
        const k2 = dg + b21 * k2a - b22 * k2b; k2b = k2a; k2a = k2;
        te = te * kTick + kick[K_TICK]; kick[K_TICK] = 0;
        wl += kWh * (nz - wl);
        let mech = k1 * gK1 + k2 * gK2 + nzh * te + wl * whirrG;
        if (fanG > 0) { fph += fInc; if (fph >= 1) fph -= 1; const fi = fph * SINE_N, ii = fi | 0; mech += (SINE[ii] + (SINE[ii + 1] - SINE[ii]) * (fi - ii)) * fanG; }
        s += intake * gInt + mech * gMech;
        if (tap) { tap[1][i] = intake * gInt; tap[2][i] = mech * gMech; }

        if (hasTurbo) {
          tl += tf * tbd; const th = nz - tl - 0.07 * tbd; tbd += tf * th;
          wph += wInc; if (wph >= 1) wph -= 1;
          const fi = wph * SINE_N, ii = fi | 0;
          let tu = tbd * gWhN + (SINE[ii] + (SINE[ii + 1] - SINE[ii]) * (fi - ii)) * gWhT;
          if (bovOn) {
            bovS += 0.02 * (bovTo - bovS);
            bl += bf * bb; const bh = nz - bl - 0.9 * bb; bb += bf * bh;
            tu += bb * bovS * 0.28;
          }
          s += tu * gTurbo;
          if (tap) tap[3][i] = tu * gTurbo;
        }
        if (hasSc) {
          sph += sInc; if (sph >= 1) sph -= 1; const fi = sph * SINE_N, ii = fi | 0; s += (SINE[ii] + (SINE[ii + 1] - SINE[ii]) * (fi - ii)) * scG * gTurbo;
          // (a blower's second harmonic, when its spec has one: the hard edge of a Roots or twin-screw whine)
          if (scH2 > 0) { let p2 = sph * 2; if (p2 >= 1) p2 -= 1; const f2 = p2 * SINE_N, i2 = f2 | 0; s += (SINE[i2] + (SINE[i2 + 1] - SINE[i2]) * (f2 - i2)) * scG * scH2 * gTurbo; }
        }
        if (hasWhine) {
          q1 += q1Inc; if (q1 >= 1) q1 -= 1; q2 += q2Inc; if (q2 >= 1) q2 -= 1;
          let fi = q1 * SINE_N, ii = fi | 0; const s1 = SINE[ii] + (SINE[ii + 1] - SINE[ii]) * (fi - ii);
          fi = q2 * SINE_N; ii = fi | 0; const s2 = SINE[ii] + (SINE[ii + 1] - SINE[ii]) * (fi - ii);
          // a gear tooth is not a sine: the squared term adds the octave above
          const wh = (s1 + 0.5 * (2 * s1 * s1 - 1)) * wA1 + (s2 + 0.4 * (2 * s2 * s2 - 1)) * wA2;
          s += wh * gWhine;
          if (tap) tap[4][i] = wh * gWhine;
        }
      }
      if (tap) tap[0][i] = ex * gExh;
      out[i] = s;
    }

    this.ph = ph; this.ev = ev; this.w = w; this.e = e; this.b = b; this.m = mm; this.seed = seed; this.hpN = hpN; this.pe = pe; this.rz = rz;
    this.hi = hi; this.hlp = hlp; this.pi = pi; this.plp = plp; this.m1 = m1; this.m2 = m2; this.rad = rad; this.c1 = c1; this.c2 = c2;
    this.is = is; this.isa = isa; this.isb = isb; this.ks = ks; this.xga = xga; this.xgb = xgb; this.i1a = i1a; this.i1b = i1b; this.i2a = i2a; this.i2b = i2b; this.hl = hl; this.hb = hb;
    this.k1a = k1a; this.k1b = k1b; this.k2a = k2a; this.k2b = k2b; this.te = te; this.wl = wl; this.fph = fph;
    this.tl = tl; this.tb = tbd; this.wph = wph; this.bl = bl; this.bb = bb; this.bovS = bovS; this.sph = sph; this.q1 = q1; this.q2 = q2;
  }
}

// ================================================================================================
// The electric car
// ================================================================================================
// An electric motor does make a note: magnetic forces whine at multiples of its turning speed, the
// reduction gear sings, and the inverter's switching sits high above with sidebands that spread as
// the motor speeds up. How loud all of it is follows torque, not speed, so lifting off goes quiet
// (with a little left for the regeneration) and standing still is silent.
export class EvVoice {
  constructor(spec, sr, opts = {}) {
    this.spec = spec; this.sr = sr; this.lite = !!opts.lite; this.dtc = CTL / sr; this.ctl = 0; this.tap = null;
    const E = spec.ev, list = [];
    // (the fourth number is which motor a partial belongs to: 0 the rear (left) one, 1 the other, 2 the reduction gear,
    // 3 the inverter, 4 the rear right one, heard only on a torque-vectoring car: see setState)
    for (const t of E.tones) { list.push([t.order, t.level, 0, 0]); if (!this.lite) list.push([t.order * E.second, t.level * 0.6, 0, 1]); }
    list.push([E.gear.order, E.gear.level, 0, 2]);
    // inverter: fixed carrier, sidebands either side at twice and four times the electrical frequency
    if (!this.lite) for (const k of [-4, -2, 2, 4]) list.push([k * E.carrier.poles, E.carrier.level * (Math.abs(k) === 2 ? 1 : 0.5), E.carrier.hz, 3]);
    if (!this.lite) for (const t of E.tones) if (t.order === 8 || t.order === 24) list.push([t.order, t.level * 0.6, 0, 4]);
    this.np = list.length;
    this.ord = new Float32Array(this.np); this.lvl = new Float32Array(this.np); this.base = new Float32Array(this.np);
    this.ph = new Float32Array(this.np); this.inc = new Float32Array(this.np); this.amp = new Float32Array(this.np); this.grp = new Uint8Array(this.np);
    list.forEach((p, i) => { this.ord[i] = p[0]; this.lvl[i] = p[1]; this.base[i] = p[2]; this.grp[i] = p[3]; this.ph[i] = (i * 0.37) % 1; });
    // torque vectoring: each motor's share of the work (front, rear left, rear right) and the outside rear wheel's
    // extra speed in a bend; tvOn stays 0 on any other car, and then nothing below changes its sound
    this.tvOn = 0; this.tvG = [1, 1, 1, 1, 0]; this.tvF = 0.5; this.tvL = 0.25; this.tvR = 0.25; this.tvW = 0;
    this.t = { rpm: 0, thr: 0, cam: 0 };
    this.rpm = 0; this.tq = 0; this.A = 0; this.ATo = 0;
    this.seed = (spec.seed * 40503) | 0 || 7; this.nl = 0; this.nb = 0; this.nf = 0.2; this.nG = 0;
    this.kRpm = 1 - Math.exp(-this.dtc / 0.03); this.kTq = 1 - Math.exp(-this.dtc / 0.05);
    this.viewG = 1; this.viewTo = 1; this.kView = 1 - Math.exp(-this.dtc / 0.09);
  }

  setState(st) {
    const t = this.t;
    if (st.rpm !== undefined) t.rpm = clamp(+st.rpm || 0, 0, 1.05);
    if (st.thr !== undefined) t.thr = clamp(+st.thr || 0, 0, 1);
    if (st.cam !== undefined) { t.cam = clamp(st.cam | 0, 0, 3); this.viewTo = [1, 0.8, 1.05, 1.25][t.cam]; }
    if (st.regen !== undefined) t.regen = clamp(+st.regen || 0, 0, 1);
    if (st.tvF !== undefined) { this.tvOn = 1; t.tvF = clamp(+st.tvF || 0, 0, 1); t.tvL = clamp(+st.tvL || 0, 0, 1); t.tvR = clamp(+st.tvR || 0, 0, 1); t.tvW = clamp(+st.tvW || 0, -0.06, 0.06); }
  }

  prime(st) { this.setState(st); this.rpm = this.t.rpm; this.tq = this.t.thr; this.viewG = this.viewTo; this.ctl = 0; this._control(); this.A = this.ATo; this.ctl = 0; }

  // A torque-vectoring car: each motor sings as hard as it works, the front motor's note (the beat 2.7% above)
  // swelling as the drive goes forward, and the two rear motors parting in a bend, the outside one a little
  // higher (its wheel turns faster) and louder when it is the one pushing. Even shares sound as the car always did.
  _tvMix() {
    const t = this.t, k = this.kTq;
    this.tvF += (t.tvF - this.tvF) * k; this.tvL += (t.tvL - this.tvL) * k; this.tvR += (t.tvR - this.tvR) * k; this.tvW += (t.tvW - this.tvW) * k;
    const g = this.tvG;
    g[1] = clamp(Math.sqrt(this.tvF / 0.5), 0, 1.5);
    g[0] = 0.78 * clamp(Math.sqrt(this.tvL / 0.25), 0, 1.6);
    g[4] = clamp(Math.sqrt(this.tvR / 0.25), 0, 1.6);
  }

  _control() {
    const sp = this.spec, sr = this.sr, t = this.t;
    this.rpm += (t.rpm - this.rpm) * this.kRpm;
    const rpm = this.rpm;
    // slowing on the motor: as hard as the physics says it regenerates (t.regen, 0..1: about 0.6 lifting off at
    // Normal, up to 1 blended into hard braking), so the whine swells on a lift and falls away with the speed;
    // a car the physics does not report on keeps the old small amount
    const regen = t.thr >= 0.05 ? 0 : t.regen !== undefined ? 0.05 + 0.55 * t.regen : 0.15 * Math.min(1, rpm * 5);
    this.tq += (Math.max(t.thr, regen) - this.tq) * this.kTq;
    const moving = clamp((rpm - 0.022) / 0.03, 0, 1);
    this.viewG += (this.viewTo - this.viewG) * this.kView;
    this.ATo = (0.1 + 0.9 * Math.pow(this.tq, 0.8)) * moving * this.viewG;
    const motorHz = (rpm * sp.redline) / 60;
    const low = 1 - clamp(rpm * 2.2, 0, 0.75);                             // the inverter is plainest when pulling away
    const tv = this.tvOn, grp = this.grp, tvG = this.tvG;
    if (tv) this._tvMix();
    for (let p = 0; p < this.np; p++) {
      // (torque vectoring: the rear left motor runs slower than the middle of the car in a left-hand bend, the right faster)
      const gp = grp[p], f = Math.abs(this.base[p] + this.ord[p] * motorHz * (tv && gp === 0 ? 1 - this.tvW : tv && gp === 4 ? 1 + this.tvW : 1));
      this.inc[p] = f / sr;
      let a = this.lvl[p] * clamp((f - 40) / 140, 0, 1) * clamp((9800 - f) / 3200, 0, 1);
      if (f > 2500) a *= Math.sqrt(2500 / f);                               // the higher a tone, the less of it is wanted
      if (this.base[p] > 0) a *= low;
      if (f > sr * 0.42) a = 0;
      if (gp === 4 || (tv && gp < 2)) a *= tv ? tvG[gp] : 0;
      this.amp[p] = a * 0.2;
    }
    this.nf = svfF(700 + 3200 * rpm, sr);
    this.nG = 0.25 * Math.pow(rpm, 1.5) * (0.4 + 0.6 * this.tq);
  }

  render(out, n) {
    let i = 0;
    while (i < n) {
      if (this.ctl <= 0) { this._control(); this.ctl = CTL; }
      const m = Math.min(n - i, this.ctl);
      const ph = this.ph, inc = this.inc, amp = this.amp, np = this.np, ATo = this.ATo, nf = this.nf, nG = this.nG;
      let A = this.A, seed = this.seed, nl = this.nl, nb = this.nb;
      for (let j = i, end = i + m; j < end; j++) {
        A += 0.004 * (ATo - A);
        let s = 0;
        for (let p = 0; p < np; p++) {
          let x = ph[p] + inc[p]; if (x >= 1) x -= 1; ph[p] = x;
          const fi = x * SINE_N, ii = fi | 0;
          s += (SINE[ii] + (SINE[ii + 1] - SINE[ii]) * (fi - ii)) * amp[p];
        }
        seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
        const nz = seed * 4.656612873077393e-10;
        nl += nf * nb; const nh = nz - nl - 1.2 * nb + DN; nb += nf * nh;      // air and bearings: a soft rush under the tones
        out[j] = (s + nb * nG) * A;
      }
      this.A = A; this.seed = seed; this.nl = nl; this.nb = nb;
      i += m; this.ctl -= m;
    }
    if (this.tap) for (let j = 0; j < n; j++) this.tap[0][j] = out[j];
  }
}

export const makeEngine = (spec, sr, opts) => (spec.kind === 'ev' ? new EvVoice(spec, sr, opts) : new EngineVoice(spec, sr, opts));

// ================================================================================================
// Tyres, wind, knocks
// ================================================================================================
// Surfaces, in the game's order (surfaces.js): grass, gravel, tarmac, mud, snow, sand.
//   hz, hzV   centre of the rolling hiss, and how it rises with speed          q     how narrow it is
//   hiss      how loud it gets          rumble   the low roar under it
//   tick..    loose stones and clods hitting the underside: [per m/s of speed, per m/s of sliding, low Hz, high Hz, ring time, loudness]
//   grain..   the surface's own texture, as bursts: [per second at rest, per m/s, low Hz, high Hz, q, attack, decay, loudness]
//   crunch..  the ground giving under the tread, as a dense rain of tiny clicks (thousands a second, each a
//             burst of noise a fraction of a millisecond long, every one its own size, length and brightness):
//             [clicks per m/s of wheel speed, extra per m/s of sliding, loudness, shortest and longest click (ms),
//              dullest and brightest click (Hz), how uneven their sizes are (1 even .. 4 a few big among many small),
//              the thud under them, how quickly the density wanders (s): patches of bigger and smaller stones]
//   A smooth band of hiss (the old way of doing loose ground) reads as running water; the clicks are what
//   make it gravel. Tarmac has no clicks: it is the smooth roar.
const GROUND = [
  { hz: 2400, hzV: 10, q: 1.6, hiss: 0.07, rumble: 0.3, tick: [0.25, 0.5, 260, 700, 0.012, 0.22], grain: [1.5, 0.28, 2400, 4600, 1.4, 0.02, 0.07, 0.18],
    crunch: [16, 60, 0.5, 0.9, 4, 700, 2600, 2, 0.35, 0.05] },                                                                                             // grass: swish, soft bumps
  { hz: 1500, hzV: 12, q: 1.3, hiss: 0.05, rumble: 0.42, tick: [0.35, 6, 1800, 6000, 0.0022, 0.24], grain: null,
    crunch: [110, 260, 1, 0.12, 0.9, 2600, 9500, 3, 0.55, 0.03] },                                                                                           // gravel: a crunch of stones
  { hz: 850, hzV: 14, q: 1.3, hiss: 0.11, rumble: 0.3, tick: null, grain: null, crunch: null },                                                             // tarmac: a smooth roar (and squeal, below)
  { hz: 700, hzV: 6, q: 1.2, hiss: 0.05, rumble: 0.45, tick: [0.6, 2, 240, 900, 0.005, 0.3], grain: [1.2, 0.34, 340, 950, 0.33, 0.012, 0.075, 0.42],
    crunch: [45, 130, 0.55, 0.5, 2.8, 450, 1800, 2, 0.8, 0.04] },                                                                                            // mud and dirt: soft, wet, squelchy
  { hz: 1400, hzV: 8, q: 1.7, hiss: 0.05, rumble: 0.3, tick: null, grain: [3, 1.7, 1200, 2700, 0.7, 0.002, 0.014, 0.4],
    crunch: [55, 150, 0.6, 0.6, 2.2, 1300, 3600, 2.5, 0.4, 0.06] },                                                                                          // snow: a squeaky packed crunch
  { hz: 1800, hzV: 20, q: 1.7, hiss: 0.1, rumble: 0.35, tick: [0.3, 1.2, 1500, 3200, 0.002, 0.12], grain: null,
    crunch: [260, 400, 0.6, 0.08, 0.35, 3200, 8000, 1, 0.2, 0.012] },                                                                                         // sand: a hissing shuffle
];
const NT = 4;                           // stones that can ring at once

export class CarFx {
  constructor(fx, sr, seed = 99) {
    this.fx = fx || { squeal: 0.3, mass: 1, openTop: false }; this.sr = sr; this.dtc = CTL / sr; this.ctl = 0;
    this.rng = new Rng(seed ^ 0x2545f491);
    this.seedA = (seed * 7919 + 17) | 0 || 3; this.seedB = (seed * 104729 + 29) | 0 || 5;
    this.t = { speed: 0, slip: 0, sf: 2, sr: 2, air: 0, brake: 0, wet: 0, cam: 0, windVol: 1 };
    this.speed = 0; this.slip = 0; this.w = new Float32Array(6);
    // rolling hiss, low roar, wind, texture bursts
    this.al = 0; this.ab = 0; this.af = 0.3; this.aq = 1; this.gHiss = 0;
    this.r1 = 0; this.r2 = 0; this.kr = onePole(240, sr); this.gRum = 0;
    this.wl = 0; this.wb = 0; this.wf = 0.1; this.gWind = 0; this.gust = 0; this.gustTo = 0; this.gustT = 0;
    // crunch: a stream of tiny clicks (see GROUND). seedC decides when each comes and what it is like;
    // ce is the loudness of the click now sounding, cl1 its brightness filter, chp takes the lows out, cb the thud
    this.seedC = (seed * 69069 + 1) | 0 || 7; this.pC = 0; this.aC = 0; this.expC = 2; this.dC0 = 0.9; this.dC1 = 0.99; this.bC0 = 0.3; this.bC1 = 0.9;
    this.ce = 0; this.ckd = 0.9; this.cl1 = 0; this.ckb = 0.5; this.cnorm = 1; this.chp = 0; this.kChp = onePole(260, sr); this.cb = 0; this.kCb = onePole(60, sr); this.gCb = 0;
    this.clump = 1; this.clumpTo = 1; this.clumpT = 0;
    // scrub: the tyre dragged sideways over loose ground, a rough grinding that comes and goes in little jerks
    this.sc1 = 0; this.sc2 = 0; this.kSc1 = onePole(2200, sr); this.kSc2 = onePole(300, sr); this.se = 0; this.seTo = 0; this.gScrub = 0;
    this.gl = 0; this.gb = 0; this.gf = 0.2; this.gq = 1; this.gT = 0; this.ge = 0; this.kGa = 0.01; this.kGd = 0.999;
    // tyre squeal: two narrow resonances that wander
    this.s1l = 0; this.s1b = 0; this.s2l = 0; this.s2b = 0; this.s1f = 0.1; this.s2f = 0.13; this.gSq = 0; this.sqW = 0; this.sqWTo = 0; this.sqT = 0;
    // brakes
    this.kl = 0; this.kb2 = 0; this.kf = svfF(2350, sr); this.gBrk = 0;
    // stones
    this.ta1 = new Float32Array(NT); this.ta2 = new Float32Array(NT); this.ty1 = new Float32Array(NT); this.ty2 = new Float32Array(NT);
    this.tk = new Float32Array(NT); this.tpL = new Float32Array(NT); this.tpR = new Float32Array(NT); this.tn = 0;
    for (let k = 0; k < NT; k++) { const p = (k + 0.5) / NT; this.tpL[k] = Math.cos(p * Math.PI * 0.5); this.tpR[k] = Math.sin(p * Math.PI * 0.5); }
    this.sched = [];
    // thumps: a falling low note each, plus a burst of dull noise
    this.thPh = new Float32Array(2); this.thF = new Float32Array(2); this.thFTo = new Float32Array(2); this.thE = new Float32Array(2); this.thKd = new Float32Array(2); this.thN = 0;
    this.kThF = 1 - Math.exp(-1 / (0.03 * sr));
    // the last landing's size and how long ago it was (s): a bounce straight after a landing, or a car that
    // touches the 9 cm line twice on its way down, does not thump again at full size
    this.lh = 0; this.lt = 9;
    this.bn = 0; this.kBn = decayK(0.06, sr); this.bn1 = 0; this.kBl = onePole(420, sr);
    // crunch of a hit: wide noise chopped into grains
    this.cr = 0; this.kCr = decayK(0.1, sr); this.crl = 0; this.crb = 0; this.crf = svfF(1300, sr);
    // what the camera hears of it
    this.lpL = 0; this.lpR = 0; this.kLp = onePole(9000, sr); this.kLpTo = this.kLp; this.vg = 1; this.vgTo = 1; this.vr = 1; this.vrTo = 1;
    this.k60 = 1 - Math.exp(-this.dtc / 0.06); this.k90 = 1 - Math.exp(-this.dtc / 0.09);
  }

  setState(st) {
    const t = this.t;
    if (st.speed !== undefined) t.speed = Math.abs(+st.speed || 0);
    if (st.slip !== undefined) t.slip = Math.max(0, +st.slip || 0);
    if (st.sf !== undefined) t.sf = clamp(st.sf | 0, 0, 5);
    if (st.sr !== undefined) t.sr = clamp(st.sr | 0, 0, 5);
    if (st.air !== undefined) t.air = st.air ? 1 : 0;
    if (st.brake !== undefined) t.brake = clamp(+st.brake || 0, 0, 1);
    if (st.wet !== undefined) t.wet = clamp(+st.wet || 0, 0, 1);
    if (st.windVol !== undefined) t.windVol = clamp(+st.windVol || 0, 0, 1);      // the Wind setting (profile.settings.wind)
    if (st.cam !== undefined) {
      t.cam = clamp(st.cam | 0, 0, 3);
      const inside = t.cam === 3 && !this.fx.openTop;
      this.kLpTo = onePole(inside ? 1900 : t.cam === 1 ? 7000 : 9000, this.sr);
      this.vgTo = inside ? 0.85 : t.cam === 1 ? 0.8 : 1;       // tyres and stones through the shell
      this.vrTo = inside ? 1.7 : 1;                             // ...but the floor booms
    }
  }

  // A stone, clod or bolt: one short ringing note. \`at\` control steps from now.
  _tick(hz, ring, amp, at = 0) {
    if (at > 0) { if (this.sched.length < 24) this.sched.push([at, hz, ring, amp]); return; }
    // the last ringer is kept for low thuds, the others take the high clicks in turn
    let k = NT - 1;
    if (hz >= 800) { k = this.tn; this.tn = (k + 1) % (NT - 1); }
    const th = (TAU * Math.min(hz, this.sr * 0.3)) / this.sr, r = Math.exp(-1 / (ring * this.sr));
    this.ta1[k] = 2 * r * Math.cos(th); this.ta2[k] = r * r;
    // whatever was still ringing here stops: carried over to a new note it would come out at the wrong size
    this.ty1[k] = 0; this.ty2[k] = 0;
    this.tk[k] = amp * Math.sin(th);
  }

  _thump(f0, f1, seconds, amp) {
    const k = this.thN; this.thN = 1 - k;
    this.thF[k] = f0 / this.sr; this.thFTo[k] = f1 / this.sr; this.thKd[k] = decayK(seconds, this.sr);
    if (this.thE[k] < 1e-3) this.thPh[k] = 0;                    // start from zero so a new thump does not click
    this.thE[k] = Math.max(this.thE[k], amp);
  }

  // type: 'land' (wheels back on the ground), 'bottom' (suspension on its stops), 'hit' (into something)
  event(type, impact) {
    const rng = this.rng, m = this.fx.mass, loose = this.w[1] + this.w[5] + this.w[0] * 0.4 + this.w[3] * 0.5;
    if (type === 'land') {
      // impact: the speed the car meets the ground at (vehicle.js _touchdown): a hop over a bump 1-2 m/s, a jump
      // 2-6, a big one 8 and up
      let a = clamp((impact - 0.8) / 7, 0, 1);
      if (this.lt < 0.3 && a <= this.lh * 1.15) a *= 0.35;      // no bigger than one just before: quietly
      else { this.lh = a; this.lt = 0; }
      if (a < 0.03) return;
      this._thump(88 / Math.sqrt(m), 40, 0.13, 0.5 * Math.pow(a, 0.8));
      this.bn = Math.max(this.bn, 0.9 * a);
      for (let i = 0; i < 2 + a * 4; i++) this._tick(500 + 1100 * rng.next(), 0.012, 0.18 * a, 2 + ((rng.next() * 70) | 0));     // things in the car settling
      for (let i = 0; i < loose * a * 14; i++) this._tick(1500 + 3500 * rng.next(), 0.003, 0.22 * a, 1 + ((rng.next() * 160) | 0));   // a spray of stones
    } else if (type === 'bottom') {
      const a = clamp(impact / 8, 0, 1);
      if (a < 0.04) return;
      this._thump(150 / Math.sqrt(m), 62, 0.055, 0.55 * a);
      this._tick(950 + 400 * rng.next(), 0.011, 0.45 * a);                          // the bump stop itself
      this.bn = Math.max(this.bn, 0.4 * a);
    } else if (type === 'hit') {
      const a = clamp((impact - 0.6) / 9, 0, 1);
      if (a < 0.03) return;
      this._thump(115 / Math.sqrt(m), 46, 0.09, 0.5 * a);
      this.cr = Math.max(this.cr, 0.9 * a); this.bn = Math.max(this.bn, 0.6 * a);
      for (let i = 0; i < 3 + a * 6; i++) this._tick(700 + 2800 * rng.next(), 0.008 + 0.02 * rng.next(), 0.35 * a * (0.4 + rng.next()), (rng.next() * 130) | 0);   // panels and trim
    }
  }

  _control() {
    const t = this.t, sr = this.sr, dt = this.dtc, rng = this.rng, w = this.w;
    this.speed += (t.speed - this.speed) * this.k60;
    this.slip += (t.slip - this.slip) * this.k60;
    const v = this.speed, slip = this.slip, ground = t.air ? 0 : 0.5;
    this.lt += dt;
    for (let s = 0; s < 6; s++) w[s] += (((t.sf === s ? ground : 0) + (t.sr === s ? ground : 0)) - w[s]) * this.k60;
    const sN = clamp((v + 2.2 * slip) / 38, 0, 1.25);

    let hz = 0, q = 0, hiss = 0, rum = 0, sum = 0, tickRate = 0, grainRate = 0;
    for (let s = 0; s < 6; s++) {
      const ws = w[s]; if (ws < 1e-3) continue;
      const G = GROUND[s];
      hz += ws * (G.hz + G.hzV * v); q += ws * G.q; sum += ws;
      hiss += ws * G.hiss * Math.pow(sN, 0.9); rum += ws * G.rumble * sN;
      if (G.tick) tickRate += ws * (G.tick[0] * v + G.tick[1] * slip);
      if (G.grain && v > 0.4) grainRate += ws * (G.grain[0] + G.grain[1] * (v + slip));
    }
    const wet = t.wet * w[2] * 2;                                  // spray on a wet road
    if (sum > 1e-3) { hz /= sum; q /= sum; } else { hz = 1500; q = 1.5; }
    if (wet > 0) { hz += wet * 2200; hiss += wet * 0.22 * sN; }
    this.af = svfF(hz, sr); this.aq = q;
    this.gHiss = hiss * 0.55 * this.vg; this.gRum = rum * 0.6 * this.vr;

    // loose stones
    if (tickRate > 0 && rng.next() < Math.min(0.9, tickRate * dt)) {
      let pick = rng.next() * tickRate, s = 0;
      for (; s < 6; s++) { const G = GROUND[s]; if (!G.tick || w[s] < 1e-3) continue; pick -= w[s] * (G.tick[0] * v + G.tick[1] * slip); if (pick <= 0) break; }
      const T = (GROUND[Math.min(s, 5)].tick) || GROUND[1].tick, r = rng.next();
      this._tick(T[2] + (T[3] - T[2]) * rng.next(), T[4] * (0.6 + rng.next()), T[5] * (0.15 + 0.85 * r * r) * clamp(sN * 1.6, 0.25, 1));
      // a big one under the floor: a dull knock, not a ringing note (a low ringing note is a water drop)
      if (T === GROUND[1].tick && rng.next() < 0.06) this.bn = Math.max(this.bn, 0.35 * Math.min(sN, 1));
    }
    // crunch: what each kind of ground under the wheels asks for, mixed by how much of each there is
    let cRate = 0, cLoud = 0, cD0 = 0, cD1 = 0, cB0 = 0, cB1 = 0, cExp = 0, cBody = 0, cWand = 0, cSum = 0;
    const wheel = v + 0.7 * slip;                                  // how fast the tread is going over the ground
    for (let s = 0; s < 6; s++) {
      const ws = w[s], C = GROUND[s].crunch; if (ws < 1e-3 || !C) continue;
      cRate += ws * (C[0] * wheel + C[1] * slip); cLoud += ws * C[2]; cSum += ws;
      cD0 += ws * C[3]; cD1 += ws * C[4]; cB0 += ws * C[5]; cB1 += ws * C[6]; cExp += ws * C[7]; cBody += ws * C[8]; cWand += ws * C[9];
    }
    if (cSum > 1e-3 && wheel > 0.3) {
      cD0 /= cSum; cD1 /= cSum; cB0 /= cSum; cB1 /= cSum; cExp /= cSum; cBody /= cSum; cWand /= cSum;
      // the density wanders: patches of bigger and smaller stones, ruts, the tread blocks landing
      if ((this.clumpT -= dt) <= 0) { this.clumpTo = 0.35 + 1.3 * rng.next(); this.clumpT = cWand * (0.5 + rng.next()); }
      this.clump += (this.clumpTo - this.clump) * Math.min(1, dt / (cWand * 0.5));
      this.pC = Math.min(0.45, (cRate * this.clump) / sr) * 4294967296;     // chance of a click on each sample (as a 32-bit threshold)
      this.dC0 = decayK(cD0 * 0.001, sr); this.dC1 = decayK(cD1 * 0.001, sr);
      this.bC0 = onePole(cB0, sr); this.bC1 = onePole(cB1, sr); this.expC = cExp;
      // louder with speed; a click each is about the same size, more of them is louder by itself
      this.aC = cLoud * 0.24 * Math.pow(Math.min(sN, 1.1), 0.55) * this.vg * Math.min(1, cSum * 2);
      this.gCb = cBody * 2.2 * this.vr;
    } else { this.pC = 0; this.aC = 0; this.gCb = 0; }
    // scrub: sliding over loose ground; on tarmac the squeal does that job
    const loose = w[0] + w[1] + w[3] + w[4] + w[5];
    // (0.12: at 0.05 it measured under 1% of a slide's sound, i.e. not there; now about a tenth of it, under the crunch)
    this.gScrub = 0.12 * loose * clamp((slip - 1.2) / 8, 0, 1) * clamp(v / 3, 0.3, 1) * this.vg;
    this.seTo = this.gScrub * (0.15 + 0.85 * Math.pow(rng.next(), 2));
    // texture bursts
    if (grainRate > 0 && rng.next() < Math.min(0.9, grainRate * dt)) {
      let pick = rng.next() * grainRate, s = 0;
      for (; s < 6; s++) { const G = GROUND[s]; if (!G.grain || w[s] < 1e-3) continue; pick -= w[s] * (G.grain[0] + G.grain[1] * (v + slip)); if (pick <= 0) break; }
      const R = GROUND[Math.min(s, 5)].grain || GROUND[0].grain;
      this.gf = svfF(R[2] + (R[3] - R[2]) * rng.next(), sr); this.gq = R[4];
      this.kGa = 1 - Math.exp(-1 / (R[5] * sr)); this.kGd = decayK(R[6] * (0.6 + 0.8 * rng.next()), sr);
      this.gT = Math.max(this.gT, R[7] * (0.35 + 0.65 * rng.next()) * clamp(sN * 2.2, 0.2, 1) * this.vg);
    }
    for (let i = this.sched.length - 1; i >= 0; i--) { const e = this.sched[i]; if (--e[0] <= 0) { this.sched.splice(i, 1); this._tick(e[1], e[2], e[3]); } }

    // squeal: tarmac only, once the tyres are properly sliding; higher and louder the faster they slide
    const sq = w[2] * 2 * clamp((slip - 2.1) / 7, 0, 1) * (1 - 0.7 * t.wet) * clamp(v / 4, 0, 1);
    if ((this.sqT -= dt) <= 0) { this.sqWTo = rng.tri(); this.sqT = 0.05 + 0.12 * rng.next(); }
    this.sqW += (this.sqWTo - this.sqW) * 0.03;
    const sqHz = (820 + 55 * Math.min(slip, 15)) * (1 + 0.025 * this.sqW);
    this.s1f = svfF(sqHz, sr); this.s2f = svfF(sqHz * 1.34, sr);
    this.gSq += (Math.pow(sq, 1.2) * 0.012 * this.vg - this.gSq) * this.k60;
    if (this.gSq < 1e-6) this.gSq = 0;

    // brakes squeak on old cars as they come to rest
    const brk = this.fx.squeal * (t.air ? 0 : 1) * clamp((t.brake - 0.25) * 2, 0, 1) * clamp(v / 0.6, 0, 1) * clamp((4.5 - v) / 3, 0, 1);
    this.gBrk += (brk * 0.006 - this.gBrk) * this.k90;
    if (this.gBrk < 1e-7) this.gBrk = 0;

    // wind over the body: a soft, dull rush under everything else. It grows with the square of speed but is
    // capped (it never gets loud), its gusts are slow, and there is nothing in it above about 700 Hz: the hiss
    // and whistle of the old wind were what made it tiring. The Wind setting (t.windVol) turns it down.
    if ((this.gustT -= dt) <= 0) { this.gustTo = rng.tri(); this.gustT = 0.9 + 1.8 * rng.next(); }
    this.gust += (this.gustTo - this.gust) * 0.0012;
    // (level and tone both stop growing at 160 km/h, 44.5 m/s: faster than that it is no louder or brighter)
    const vw = Math.min(v, 44.5), wN = vw / 55, cam = t.cam;
    this.wf = svfF(200 + 9 * vw, sr);
    this.gWind = 0.16 * wN * wN * (1 + 0.15 * this.gust) * (cam === 3 && !this.fx.openTop ? 0.55 : cam === 2 ? 1.2 : 1) * t.windVol;

    this.kLp += (this.kLpTo - this.kLp) * this.k90; this.vg += (this.vgTo - this.vg) * this.k90; this.vr += (this.vrTo - this.vr) * this.k90;
    if (this.gT < 1e-9) this.gT = 0;
    if (this.bn < 1e-9) this.bn = 0;
    if (this.cr < 1e-9) this.cr = 0;
    for (let k = 0; k < 2; k++) if (this.thE[k] < 1e-6) this.thE[k] = 0;
  }

  // Add n samples to L and R.
  render(L, R, n, gain = 1) {
    let i = 0;
    while (i < n) {
      if (this.ctl <= 0) { this._control(); this.ctl = CTL; }
      const m = Math.min(n - i, this.ctl);
      this._run(L, R, i, m, gain);
      i += m; this.ctl -= m;
    }
  }

  _run(L, R, i0, m, gain) {
    let sA = this.seedA, sB = this.seedB;
    let al = this.al, ab = this.ab, r1 = this.r1, r2 = this.r2, wl = this.wl, wb = this.wb, gl = this.gl, gb = this.gb, gT = this.gT, ge = this.ge;
    const af = this.af, aq = this.aq, gHiss = this.gHiss, kr = this.kr, gRum = this.gRum, wf = this.wf, gWind = this.gWind, gf = this.gf, gq = this.gq, kGa = this.kGa, kGd = this.kGd;
    const sqOn = this.gSq > 0 || Math.abs(this.s1b) > 1e-6, gSq = this.gSq, s1f = this.s1f, s2f = this.s2f;
    let s1l = this.s1l, s1b = this.s1b, s2l = this.s2l, s2b = this.s2b;
    const brkOn = this.gBrk > 0 || Math.abs(this.kb2) > 1e-6, gBrk = this.gBrk, kf = this.kf; let kl = this.kl, kb2 = this.kb2;
    const thPh = this.thPh, thF = this.thF, thFTo = this.thFTo, thE = this.thE, thKd = this.thKd, kThF = this.kThF;
    const th0 = thE[0] > 0, th1 = thE[1] > 0;
    let bn = this.bn, bn1 = this.bn1, cr = this.cr, crl = this.crl, crb = this.crb; const kBn = this.kBn, kBl = this.kBl, kCr = this.kCr, crf = this.crf;
    const ta1 = this.ta1, ta2 = this.ta2, ty1 = this.ty1, ty2 = this.ty2, tk = this.tk, tpL = this.tpL, tpR = this.tpR;
    let lpL = this.lpL, lpR = this.lpR; const kLp = this.kLp;
    let sC = this.seedC, ce = this.ce, ckd = this.ckd, ckb = this.ckb, cl1 = this.cl1, chp = this.chp, cb = this.cb;
    const pC = this.pC, aC = this.aC, expC = this.expC, dC0 = this.dC0, dC1 = this.dC1, bC0 = this.bC0, bC1 = this.bC1, kChp = this.kChp, kCb = this.kCb, gCb = this.gCb;
    let se = this.se, sc1 = this.sc1, sc2 = this.sc2; const seTo = this.seTo, gScrub = this.gScrub, kSc1 = this.kSc1, kSc2 = this.kSc2;

    for (let i = i0, end = i0 + m; i < end; i++) {
      sA = (Math.imul(sA, 1664525) + 1013904223) | 0; const n1 = sA * 4.656612873077393e-10;
      sB = (Math.imul(sB, 22695477) + 1) | 0; const n2 = sB * 4.656612873077393e-10;
      // rolling hiss and the roar under it
      al += af * ab; const ah = n1 - al - aq * ab + DN; ab += af * ah;
      r1 += kr * (n2 - r1); r2 += kr * (r1 - r2);
      let s = ab * gHiss + r2 * gRum;
      // wind: the low-pass side of the filter, so it is a soft rush with no hiss on top
      wl += wf * wb; const wh = n2 - wl - 1.4 * wb + DN; wb += wf * wh;
      s += wl * gWind;
      // crunch: now and then (pC), a click starts: its size (mostly small, now and then a big stone), how
      // quickly it dies (a fraction of a millisecond to a few), and how bright it is, all drawn at random
      if (pC > 0) {
        sC = (Math.imul(sC, 1103515245) + 12345) | 0;
        if ((sC >>> 0) < pC) {
          sC = (Math.imul(sC, 1103515245) + 12345) | 0; const u1 = (sC >>> 0) / 4294967296;
          sC = (Math.imul(sC, 1103515245) + 12345) | 0; const u2 = (sC >>> 0) / 4294967296;
          sC = (Math.imul(sC, 1103515245) + 12345) | 0; const u3 = (sC >>> 0) / 4294967296;
          ckb = bC0 + (bC1 - bC0) * u3; ckd = dC0 + (dC1 - dC0) * u2;
          ce += aC * Math.pow(u1, expC) * Math.sqrt((2 - ckb) / ckb) * (1.4 - 0.6 * u2);      // (a dull click is filtered quieter: made up; a long one is a little softer)
        }
      }
      if (ce > 1e-7) {
        ce *= ckd;
        cl1 += ckb * (n1 * ce - cl1);
        chp += kChp * (cl1 - chp);
        s += cl1 - chp;
        cb += kCb * (ce - cb);
        s += r2 * cb * gCb;                                       // the thud under the clicks: the low roar, pushed by them
      } else ce = 0;
      // scrub: grinding noise in little jerks
      if (gScrub > 0 || se > 1e-6) {
        se += 0.01 * (seTo - se);
        sc1 += kSc1 * (n2 - sc1); sc2 += kSc2 * (sc1 - sc2);
        s += (sc1 - sc2) * se;
      }
      // squelch, crunch, swish
      gT *= kGd; ge += kGa * (gT - ge);
      gl += gf * gb; const gh = n2 - gl - gq * gb + DN; gb += gf * gh;
      s += gb * ge;
      if (sqOn) {
        s1l += s1f * s1b; const h1 = n1 - s1l - 0.035 * s1b; s1b += s1f * h1;
        s2l += s2f * s2b; const h2 = n2 - s2l - 0.045 * s2b; s2b += s2f * h2;
        s += (s1b + 0.55 * s2b) * gSq;
      }
      if (brkOn) { kl += kf * kb2; const kh = n1 - kl - 0.012 * kb2; kb2 += kf * kh; s += kb2 * gBrk; }
      // thumps
      if (th0) { let p = thPh[0] + thF[0]; if (p >= 1) p -= 1; thPh[0] = p; thF[0] += (thFTo[0] - thF[0]) * kThF; thE[0] *= thKd[0]; const fi = p * SINE_N, ii = fi | 0; s += (SINE[ii] + (SINE[ii + 1] - SINE[ii]) * (fi - ii)) * thE[0]; }
      if (th1) { let p = thPh[1] + thF[1]; if (p >= 1) p -= 1; thPh[1] = p; thF[1] += (thFTo[1] - thF[1]) * kThF; thE[1] *= thKd[1]; const fi = p * SINE_N, ii = fi | 0; s += (SINE[ii] + (SINE[ii + 1] - SINE[ii]) * (fi - ii)) * thE[1]; }
      bn *= kBn; bn1 += kBl * (n1 - bn1); s += bn1 * bn * 1.6;
      if (cr > 0) { cr *= kCr; crl += crf * crb; const ch = n1 - crl - 0.9 * crb; crb += crf * ch; s += crb * cr * (0.25 + 0.75 * Math.abs(n2)) * 0.3; }
      // stones, each a little to one side
      let sl = s, sr2 = s;
      for (let k = 0; k < NT; k++) {
        const y = tk[k] + ta1[k] * ty1[k] - ta2[k] * ty2[k] + DN; tk[k] = 0; ty2[k] = ty1[k]; ty1[k] = y;
        sl += y * tpL[k]; sr2 += y * tpR[k];
      }
      lpL += kLp * (sl - lpL); lpR += kLp * (sr2 - lpR);
      L[i] += lpL * gain; R[i] += lpR * gain;
    }
    this.seedA = sA; this.seedB = sB; this.al = al; this.ab = ab; this.r1 = r1; this.r2 = r2; this.wl = wl; this.wb = wb; this.gl = gl; this.gb = gb; this.gT = gT; this.ge = ge;
    this.s1l = s1l; this.s1b = s1b; this.s2l = s2l; this.s2b = s2b; this.kl = kl; this.kb2 = kb2;
    this.bn = bn; this.bn1 = bn1; this.cr = cr; this.crl = crl; this.crb = crb; this.lpL = lpL; this.lpR = lpR;
    this.seedC = sC; this.ce = ce; this.ckd = ckd; this.ckb = ckb; this.cl1 = cl1; this.chp = chp; this.cb = cb; this.se = se; this.sc1 = sc1; this.sc2 = sc2;
  }
}

// ================================================================================================
// Loudness
// ================================================================================================
// Every engine is played, unheard, through the same stretch of a drive (foot down, from 40% to 92%
// of its revs) and its loudness measured, so that a V8 and a kei car come out of the speakers at the
// same level before the things that should make one louder (an open exhaust) are applied on top.
// A run through the revs, not one speed: at any single speed an engine may be sitting on one of its
// pipe's resonances, or between two.
// The measurement leans on the middle of the range, where ears do: a deep rumble measures quieter
// than a rasp of the same size, as it sounds.
const REF_RMS = 0.085;
const CAL_STATE = { rpm: 0.4, thr: 1, gear: 3, shiftT: 0, cam: 0, speed: 25 };
const CAL_TOP = 0.92;
export class Calibration {
  constructor(spec, sr, lite) {
    this.voice = makeEngine(spec, sr, { lite, seed: 0xca11b });
    this.voice.prime(CAL_STATE);
    this.skip = Math.round(0.1 * sr); this.left = Math.round(0.5 * sr); this.total = this.left;
    this.sum = 0; this.hp = 0; this.kHp = onePole(160, sr); this.buf = new Float32Array(2048); this.gain = 0;
  }

  // Do a little of the work; true once the answer (this.gain) is ready.
  step(maxSamples) {
    while (maxSamples > 0 && this.left > 0) {
      const n = Math.min(512, maxSamples, this.skip > 0 ? this.skip : this.left);
      if (this.skip <= 0) this.voice.setState({ rpm: CAL_STATE.rpm + (CAL_TOP - CAL_STATE.rpm) * (1 - this.left / this.total) });
      this.voice.render(this.buf, n);
      if (this.skip > 0) this.skip -= n;
      else {
        let hp = this.hp, sum = 0; const k = this.kHp, b = this.buf;
        for (let i = 0; i < n; i++) { hp += k * (b[i] - hp); const y = b[i] - hp; sum += y * y; }
        this.hp = hp; this.sum += sum; this.left -= n;
      }
      maxSamples -= n;
    }
    if (this.left > 0) return false;
    const rms = Math.sqrt(this.sum / this.total);
    this.gain = clamp(REF_RMS / Math.max(rms, 1e-5), 0.02, 12) * Math.pow(10, (this.voice.spec.levelDb || 0) / 20);
    return true;
  }
}
export function calibrate(spec, sr, lite = false) { const c = new Calibration(spec, sr, lite); while (!c.step(1 << 20)); return c.gain; }

// ================================================================================================
// Voices and the mixer
// ================================================================================================
// A player's car: the full engine and everything else the car does, heard from its own camera.
class CarVoice {
  constructor(spec, sr, gain, share) {
    this.engine = makeEngine(spec, sr, { lite: false });
    this.fx = new CarFx(spec.fx, sr, spec.seed & 0xffff);
    this.calGain = gain; this.share = share; this.fade = 0; this.fadeTo = 1; this.age = 0; this.dead = false; this.kFade = 1 / (0.07 * sr);
    this.idle = spec.kind === 'ev' ? 0 : spec.idle;
  }

  set(st) {
    this.age = 0;
    if (st.share !== undefined) this.share = clamp(+st.share || 0, 0, 2);
    this.engine.setState(st); this.fx.setState(st);
    if (st.ev) for (const e of st.ev) this.fx.event(e[0], +e[1] || 0);
  }

  // nothing heard from the game for a while (it has stalled, or is loading): settle to an idle
  park() { this.engine.setState({ rpm: this.idle, thr: 0, shiftT: 0 }); this.fx.setState({ speed: 0, slip: 0, brake: 0 }); }

  render(L, R, n, tmp) {
    this.engine.render(tmp, n);
    let f = this.fade; const to = this.fadeTo, kf = this.kFade, g = this.calGain * this.share;
    for (let i = 0; i < n; i++) {
      if (f < to) { f += kf; if (f > to) f = to; } else if (f > to) { f -= kf; if (f < to) f = to; }
      const s = tmp[i] * g * f; L[i] += s; R[i] += s;
    }
    this.fx.render(L, R, n, this.share * f);
    this.fade = f;
    if (f <= 0 && to <= 0) this.dead = true;
  }
}

// Somebody else's car: exhaust only, quieter and duller with distance, to one side or the other.
class NpcVoice {
  constructor(spec, sr, gain, share) {
    this.engine = makeEngine(spec, sr, { lite: true }); this.sr = sr;
    this.calGain = gain; this.share = share; this.fade = 0; this.fadeTo = 1; this.age = 0; this.dead = false; this.kFade = 1 / (0.12 * sr);
    this.idle = spec.kind === 'ev' ? 0 : spec.idle;
    this.dist = 400; this.pan = 0; this.dop = 1; this.rpm = this.idle; this.speed = 0; this.loose = 1;
    this.g = 0; this.gL = 0; this.gR = 0; this.lp = 0; this.kLp = 0.5; this.seed = (spec.seed * 31 + 7) | 0 || 9; this.h1 = 0; this.h2 = 0; this.kh1 = onePole(2600, sr); this.kh2 = onePole(500, sr);
  }

  set(st) {
    this.age = 0;
    if (st.share !== undefined) this.share = clamp(+st.share || 0, 0, 2);
    if (st.dist !== undefined) this.dist = Math.max(0, +st.dist || 0);
    if (st.pan !== undefined) this.pan = clamp(+st.pan || 0, -1, 1);
    if (st.dop !== undefined) this.dop = clamp(+st.dop || 1, 0.8, 1.25);
    if (st.speed !== undefined) this.speed = Math.abs(+st.speed || 0);
    if (st.sr !== undefined) this.loose = (st.sr | 0) === 2 ? 0.25 : 1;
    // a car coming towards you is higher in pitch, one going away lower: shift the revs it is heard at
    const rpm = st.rpm !== undefined ? +st.rpm || 0 : this.rpm; this.rpm = rpm;
    this.engine.setState({ rpm: rpm * this.dop, thr: st.thr, gear: st.gear, shiftT: st.shiftT, speed: st.speed, boost: st.boost, cam: 0 });
  }

  park() { this.engine.setState({ rpm: this.idle, thr: 0, shiftT: 0 }); this.speed = 0; }

  render(L, R, n, tmp) {
    const d = this.dist;
    // level falls with distance and is gone by about 320 m; the air takes the top off first
    const gTo = this.fadeTo <= 0 ? 0 : Math.pow(Math.min(1, 7 / Math.max(d, 1)), 1.25) * clamp((320 - d) / 120, 0, 1);
    if (gTo < 2e-4 && this.g < 2e-4) {             // too far to hear: do not spend anything on it
      this.g = 0; this.fade = this.fadeTo;
      if (this.fadeTo <= 0) this.dead = true;
      return;
    }
    this.engine.render(tmp, n);
    this.kLp = onePole(13000 / (1 + d / 55), this.sr);
    const kLp = this.kLp, base = this.calGain * this.share, pr = (this.pan + 1) * 0.25 * Math.PI, pL = Math.cos(pr) * 1.414, pR = Math.sin(pr) * 1.414;
    const road = clamp(this.speed / 40, 0, 1) * this.loose * 0.5, kg = 1 / (0.1 * this.sr), kh1 = this.kh1, kh2 = this.kh2;
    let g = this.g, lp = this.lp, seed = this.seed, h1 = this.h1, h2 = this.h2, f = this.fade; const to = this.fadeTo, kf = this.kFade;
    for (let i = 0; i < n; i++) {
      g += (gTo - g) * kg * 4;
      if (f < to) { f += kf; if (f > to) f = to; } else if (f > to) { f -= kf; if (f < to) f = to; }
      seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
      const nz = seed * 4.656612873077393e-10;
      h1 += kh1 * (nz - h1); h2 += kh2 * (h1 - h2);                 // its tyres on the road
      lp += kLp * (tmp[i] * base + (h1 - h2) * road * this.share - lp);
      const s = lp * g * f; L[i] += s * pL; R[i] += s * pR;
    }
    this.g = g; this.lp = lp; this.seed = seed; this.h1 = h1; this.h2 = h2; this.fade = f;
    if (f <= 0 && to <= 0) this.dead = true;
  }
}

// Messages (the same objects the game posts to the audio thread):
//   { t: 'add', id, kind: 'car' | 'npc', spec, share }     a new voice; it starts once its loudness has been measured
//   { t: 'set', id, rpm, thr, speed, slip, sf, sr, air, gear, shiftT, brake, wet, cam, ev: [[type, impact]], share,
//               dist, pan, dop, boost }                      this frame's state (dist, pan, dop: npc only; boost: the
//                                                            physics' turbo boost 0..1, or -1 / missing: the voice's own)
//   { t: 'del', id }                                         fade out and forget
//   { t: 'master', gain }                                    whole-mix level 0..1 (the tab was hidden, the game is quitting)
export class Mixer {
  constructor(sr) {
    this.sr = sr; this.voices = new Map(); this.pending = []; this.cal = new Map();
    this.tmp = new Float32Array(256); this.master = 1; this.masterTo = 1;
    this.peak = 0; this.over = 0; this.frames = 0; this.made = 0; this.removed = 0;
  }

  handle(m) {
    if (!m) return;
    if (m.t === 'set') {
      const v = this.voices.get(m.id);
      if (v) v.set(m);
      else for (const p of this.pending) if (p.msg.id === m.id) { p.last = m; break; }
    } else if (m.t === 'add') {
      this._drop(m.id);
      this.pending.push({ msg: m, work: null, last: null });
    } else if (m.t === 'del') {
      this.pending = this.pending.filter((p) => p.msg.id !== m.id);
      const v = this.voices.get(m.id);
      if (v) v.fadeTo = 0;
    } else if (m.t === 'master') this.masterTo = clamp(+m.gain || 0, 0, 1);
  }

  _drop(id) { if (this.voices.delete(id)) this.removed++; this.pending = this.pending.filter((p) => p.msg.id !== id); }

  // Measuring a new engine's loudness is spread over a few blocks so the audio thread never stalls.
  _calibrate(budget) {
    const p = this.pending[0];
    if (!p) return;
    const lite = p.msg.kind === 'npc', key = (lite ? 'n' : 'c') + JSON.stringify(p.msg.spec);
    let gain = this.cal.get(key);
    if (gain === undefined) {
      if (!p.work) p.work = new Calibration(p.msg.spec, this.sr, lite);
      if (!p.work.step(budget)) return;
      gain = p.work.gain;
      if (this.cal.size > 64) this.cal.clear();
      this.cal.set(key, gain);
    }
    this.pending.shift();
    const share = p.msg.share === undefined ? 1 : +p.msg.share || 0;
    const v = lite ? new NpcVoice(p.msg.spec, this.sr, gain, share) : new CarVoice(p.msg.spec, this.sr, gain, share);
    if (p.last) v.set(p.last);
    this.voices.set(p.msg.id, v); this.made++;
  }

  // Finish any measuring now (tools use this; the game lets it happen a block at a time).
  flush() { while (this.pending.length) this._calibrate(1 << 24); }

  silent() { return this.voices.size === 0 && this.pending.length === 0; }

  // Fill L and R (n samples each) with the whole mix.
  render(L, R, n) {
    if (this.tmp.length < n) this.tmp = new Float32Array(n);
    for (let i = 0; i < n; i++) { L[i] = 0; R[i] = 0; }
    if (this.pending.length) this._calibrate(n * 20);
    this.frames += n;
    const asleep = this.masterTo <= 0 && this.master < 1e-4;
    if (!asleep) {
      const dt = n / this.sr;
      for (const [id, v] of this.voices) {
        if ((v.age += dt) > 0.6 && v.age - dt <= 0.6) v.park();
        v.render(L, R, n, this.tmp);
        if (v.dead) { this.voices.delete(id); this.removed++; }
      }
    } else {
      for (const [id, v] of this.voices) if (v.fadeTo <= 0) { this.voices.delete(id); this.removed++; }
    }
    // master level, then a soft ceiling: untouched below 0.6, bending over to stay under 0.95
    let ms = this.master, peak = this.peak, over = 0; const to = this.masterTo, km = 1 / (0.15 * this.sr);
    for (let i = 0; i < n; i++) {
      if (ms < to) { ms += km; if (ms > to) ms = to; } else if (ms > to) { ms -= km; if (ms < to) ms = to; }
      let l = L[i] * ms, r = R[i] * ms;
      const al = l < 0 ? -l : l, ar = r < 0 ? -r : r;
      if (al > peak) peak = al; if (ar > peak) peak = ar;
      if (al > 0.6) { over++; l = (l < 0 ? -1 : 1) * (0.6 + 0.35 * Math.tanh((al - 0.6) / 0.35)); }
      if (ar > 0.6) { r = (r < 0 ? -1 : 1) * (0.6 + 0.35 * Math.tanh((ar - 0.6) / 0.35)); }
      L[i] = l; R[i] = r;
    }
    this.master = ms; this.peak = peak; this.over += over;
  }

  // For tests and the debug overlay; reading the peak resets it.
  stats() {
    const s = { voices: this.voices.size, pending: this.pending.length, made: this.made, removed: this.removed, peak: this.peak, over: this.over, frames: this.frames, master: this.master };
    this.peak = 0; this.over = 0;
    return s;
  }
}
`,s=`// The audio-thread end of the car sounds: a thin shell around the Mixer in engineDsp.js.
//
// This file is not imported by anything. audio.js takes the text of engineDsp.js, adds the text of
// this file after it, and hands the two to the browser as one audio-worklet module (from a Blob, so
// it works the same from the dev server, from a built folder, and inside a portal's iframe).
// That is why \`Mixer\` is used here without being imported: by then it is defined just above.
/* global Mixer, AudioWorkletProcessor, registerProcessor, sampleRate */

// how many of these are alive in this audio context: the game's tests read it to prove nothing leaks
const LIVE = (globalThis.__oversteerSound = globalThis.__oversteerSound || { count: 0, made: 0 });

class OversteerSound extends AudioWorkletProcessor {
  constructor() {
    super();
    this.mixer = new Mixer(sampleRate);
    this.closing = false; this.done = false;
    LIVE.count++; LIVE.made++;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.t === 'stats') this.port.postMessage({ t: 'stats', n: m.n, live: LIVE.count, everMade: LIVE.made, sampleRate, ...this.mixer.stats() });
      else if (m.t === 'close') this.closing = true;        // finish the fades, then stop for good
      else this.mixer.handle(m);
    };
  }

  process(inputs, outputs) {
    if (this.done) return false;
    const out = outputs[0], L = out[0], R = out.length > 1 ? out[1] : null;
    try {
      if (R) this.mixer.render(L, R, L.length);
      else { if (!this.mono || this.mono.length !== L.length) this.mono = new Float32Array(L.length); this.mixer.render(L, this.mono, L.length); for (let i = 0; i < L.length; i++) L[i] = 0.5 * (L[i] + this.mono[i]); }
    } catch (err) {
      // never leave the audio thread throwing 300 times a second: say what happened and stop,
      // and the game falls back to its simple engine tone
      for (const ch of out) ch.fill(0);
      this.port.postMessage({ t: 'error', message: String((err && err.stack) || err) });
      this.done = true; LIVE.count--;
      return false;
    }
    if (this.closing && this.mixer.silent()) { this.done = true; LIVE.count--; return false; }
    return true;
  }
}
registerProcessor('oversteer-sound', OversteerSound);
`,i="oversteer-sound",n=`${t}
${s}
`;export{i as WORKLET_NAME,n as WORKLET_SOURCE};
