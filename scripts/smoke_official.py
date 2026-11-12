"""Smoke test for an official benchmark environment (run with that benchmark's venv python).
  venvs/metaworld/bin/python scripts/smoke_official.py metaworld
  venvs/ocbench/bin/python   scripts/smoke_official.py ocbench
Reports: obs/action dims, steps/s, episode length, success signal availability, scripted-policy SR (where available)."""
import sys, time, json
import numpy as np
which = sys.argv[1]; out = {"env": which}
t0 = time.time()
if which == "metaworld":
    import gymnasium as gym, metaworld
    from metaworld.policies import ENV_POLICY_MAP
    res = {}
    for name in ("reach-v3", "push-v3", "pick-place-v3", "button-press-v3"):
        env = gym.make("Meta-World/MT1", env_name=name, seed=0)
        pol = ENV_POLICY_MAP[name](); succ = 0; steps = 0; n_ep = 10; t1 = time.time()
        for ep in range(n_ep):
            ob, info = env.reset(seed=ep); done = False; ok = False
            while not done:
                a = pol.get_action(ob); ob, r, te, tr, info = env.step(a); steps += 1
                ok |= bool(info.get("success", 0)); done = te or tr
            succ += ok
        res[name] = {"obs_dim": int(ob.shape[0]), "act_dim": int(env.action_space.shape[0]), "scripted_sr": succ / n_ep,
                     "steps_per_s": round(steps / (time.time() - t1), 1), "ep_len": steps // n_ep}
        env.close()
    out["tasks"] = res
elif which == "ocbench":
    import ocbench
    res = {}
    for name in ("block-cpu-single-task1-v0", "block-cpu-lite-single-task1-v0"):
        try:
            env = ocbench.make(name); ob, info = env.reset(); steps = 0; ret = 0.0; done = False; t1 = time.time(); ok = False
            while not done and steps < 300:
                ob, r, te, tr, info = env.step(env.action_space.sample()); steps += 1; ret += r; done = te or tr
                ok |= bool(info.get("success", False))
            res[name] = {"obs_shape": list(np.shape(ob)), "act_shape": list(env.action_space.shape),
                         "random_steps_per_s": round(steps / (time.time() - t1), 1), "info_keys": sorted(info.keys())[:8]}
            env.close()
        except Exception as e:
            res[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    out["tasks"] = res
elif which == "kinetix":
    import jax, jax.numpy as jnp
    out["jax"] = {"version": jax.__version__, "devices": [str(d) for d in jax.devices()]}
    import kinetix
    out["kinetix"] = getattr(kinetix, "__file__", "?")
out["wall_s"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
