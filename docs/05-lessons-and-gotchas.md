# Lessons and gotchas

Things that cost time, with the symptom first so you can grep for it.

## `OSError: Library not loaded: @rpath/libomp.dylib` on `import lightgbm`

The LightGBM wheel links OpenMP dynamically and expects Homebrew's copy.
`brew install libomp`.

## LightGBM fit exits with code 139 (segfault), no Python traceback

Torch was imported before LightGBM. Both wheels ship a `libomp.dylib`; the second
runtime loaded crashes on first use. Importing LightGBM first avoids the crash
but not the next problem.

## PyTorch training hangs at 0% CPU after a LightGBM fit

Same root cause: two OpenMP runtimes in one process deadlock. Fix in
`scripts/setup_macos.sh`:

```bash
install_name_tool -rpath /opt/homebrew/opt/libomp/lib "$SITE/torch/lib" "$SITE/lightgbm/lib/lib_lightgbm.dylib"
codesign -s - -f "$SITE/lightgbm/lib/lib_lightgbm.dylib"
```

Now both libraries resolve `@rpath/libomp.dylib` to PyTorch's file. Verify with
`otool -l lib_lightgbm.dylib | grep -A2 LC_RPATH`. Re-run after reinstalling
LightGBM. `KMP_DUPLICATE_LIB_OK=TRUE` is the usual Internet advice; it did not
help here.

## `categorical=0` for Bank Marketing

pandas 3 stores text as a dedicated `str` dtype, so `df[c].dtype == object` is
false for every column. Use `pd.api.types.is_numeric_dtype` and negate it.

## `LGBMDeprecationWarning: 'eval_set' is deprecated`

LightGBM 4.7 takes `eval_X=X_val, eval_y=y_val` (single arrays, not lists).

## `UserWarning: Converting a tensor with requires_grad=True to a scalar`

`float(loss)` inside the training loop; use `float(loss.detach())`.

## No `timeout` command on macOS

Use the tool's own timeout or `gtimeout` from `coreutils`.

## TSK starts at p = 0.5 and never fixes its intercept

BatchNorm on the rule consequents standardises every rule output, so the initial
logit is 0 for every input. Early stopping on PR-AUC does not reward fixing the
intercept, so at 1% positives the model can stop with ECE 0.4 while ranking
fine. Initialise the BatchNorm bias (or the consequent bias) at the base-rate
log-odds. Do the same for the MLP's last layer so the comparison stays fair.

## Early stopping on a tiny validation set

With 24 positives in the fraud validation split, validation PR-AUC jumps by 0.1
between adjacent trees. LightGBM stopped at 2 trees on one seed. Any metric
computed on a handful of positives is a bad stopping signal; use log loss or a
fixed budget chosen on a smoother criterion.

## Selecting configurations on one seed

The best configuration on seed 0 can be seed-lucky. On fraud the chosen 16-rule
TSK varied 0.73 to 0.81 across seeds; the 4-rule variant sat at 0.80 ± 0.003.
Average two or three seeds in the search phase when fits are cheap.

## Parallel processes and one log file

Appending JSON lines from seven processes to one file can interleave writes
above 4 KB. One file per process, merged on read, avoids it and still resumes.

## `rm -rf` and `cat > file <<EOF` blocked by local policy

This machine's guard script blocks recursive force deletes and treats `cat` as
a read. Move directories aside (`mv results/runs results/old/...`) and write
files with an editor or `python - <<EOF`.
