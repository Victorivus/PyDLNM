#!/usr/bin/env Rscript
# generate_fixtures.R
#
# Generates CSV golden-reference fixtures from R's dlnm package.
# Run from the PyDLNM repository root:
#
#   Rscript tests/fixtures/generate_fixtures.R
#
# On Windows, Rscript may not be in PATH.  Use the full path, e.g.:
#   "C:/Program Files/R/R-4.4.2/bin/Rscript.exe" tests/fixtures/generate_fixtures.R
#
# Requirements: R >= 4.2, dlnm >= 2.4.7
#
# After running, commit all generated CSV files.  Record the R and dlnm
# versions in tests/fixtures/README.md.

suppressPackageStartupMessages({
  library(dlnm)
  library(splines)
})

cat("R version:", R.version$version.string, "\n")
cat("dlnm version:", as.character(packageVersion("dlnm")), "\n\n")

FIXTURES <- file.path("tests", "fixtures")
dir.create(FIXTURES, showWarnings = FALSE, recursive = TRUE)

save_csv <- function(x, name) {
  path <- file.path(FIXTURES, name)
  write.csv(x, path, row.names = FALSE)
  cat("  wrote:", name, "\n")
}

# ---------------------------------------------------------------------------
# 1. Synthetic data — deterministic
# ---------------------------------------------------------------------------
cat("=== Synthetic data ===\n")
set.seed(42)
n <- 200
x <- rnorm(n, mean = 20, sd = 5)
save_csv(data.frame(x = x), "synthetic_x.csv")

# Fixed coefficient vectors (generated once, reused everywhere)
# cb_ns_df4_ns_df3 has 4*3 = 12 columns
set.seed(42)
coef_12 <- rnorm(12) * 0.001
save_csv(data.frame(coef = coef_12), "coef_12.csv")
vcov_12 <- diag(12) * 0.0001   # used inline, not saved (identity-like)

# ---------------------------------------------------------------------------
# 2. onebasis — ns
# ---------------------------------------------------------------------------
cat("\n=== onebasis ===\n")

ob <- onebasis(x, fun = "ns", df = 4)
save_csv(as.data.frame(unclass(ob)), "onebasis_ns_df4.csv")

ob <- onebasis(x, fun = "ns", df = 4, intercept = TRUE)
save_csv(as.data.frame(unclass(ob)), "onebasis_ns_df4_intercept.csv")

ob <- onebasis(x, fun = "ns", knots = c(15, 20, 25))
save_csv(as.data.frame(unclass(ob)), "onebasis_ns_knots.csv")

# onebasis — bs
ob <- onebasis(x, fun = "bs", df = 5)
save_csv(as.data.frame(unclass(ob)), "onebasis_bs_df5.csv")

# onebasis — lin
ob <- onebasis(x, fun = "lin")
save_csv(as.data.frame(unclass(ob)), "onebasis_lin.csv")

# onebasis — poly
ob <- onebasis(x, fun = "poly", degree = 2)
save_csv(as.data.frame(unclass(ob)), "onebasis_poly_deg2.csv")

# onebasis — thr
ob <- onebasis(x, fun = "thr", thr.value = 20, side = "h")
save_csv(as.data.frame(unclass(ob)), "onebasis_thr_h.csv")

ob <- onebasis(x, fun = "thr", thr.value = 20, side = "l")
save_csv(as.data.frame(unclass(ob)), "onebasis_thr_l.csv")

ob <- onebasis(x, fun = "thr", thr.value = c(15, 25), side = "d")
save_csv(as.data.frame(unclass(ob)), "onebasis_thr_d.csv")

# onebasis — strata
ob <- onebasis(x, fun = "strata", df = 3)
save_csv(as.data.frame(unclass(ob)), "onebasis_strata_df3.csv")

# onebasis — integer  (use a short integer sequence as input)
lag_seq <- 0:10
ob <- onebasis(lag_seq, fun = "integer")
save_csv(as.data.frame(unclass(ob)), "onebasis_integer_lagseq.csv")

# ---------------------------------------------------------------------------
# 3. crossbasis
# ---------------------------------------------------------------------------
cat("\n=== crossbasis ===\n")

cb_ns_ns <- crossbasis(
  x, lag = 10,
  argvar = list(fun = "ns", df = 4),
  arglag = list(fun = "ns", df = 3)
)
save_csv(as.data.frame(unclass(cb_ns_ns)), "crossbasis_ns_df4_ns_df3_lag10.csv")

cb_bs_ns <- crossbasis(
  x, lag = 10,
  argvar = list(fun = "bs", df = 4),
  arglag = list(fun = "ns", df = 3)
)
save_csv(as.data.frame(unclass(cb_bs_ns)), "crossbasis_bs_df4_ns_df3_lag10.csv")

cb_lin_ns <- crossbasis(
  x, lag = 10,
  argvar = list(fun = "lin"),
  arglag = list(fun = "ns", df = 3)
)
save_csv(as.data.frame(unclass(cb_lin_ns)), "crossbasis_lin_ns_df3_lag10.csv")

cb_ns_strata <- crossbasis(
  x, lag = 10,
  argvar = list(fun = "ns", df = 4),
  arglag = list(fun = "strata", df = 2)
)
save_csv(as.data.frame(unclass(cb_ns_strata)), "crossbasis_ns_df4_strata_df2_lag10.csv")

cb_ns_ns_21 <- crossbasis(
  x, lag = c(0, 21),
  argvar = list(fun = "ns", df = 4),
  arglag = list(fun = "ns", df = 3)
)
save_csv(as.data.frame(unclass(cb_ns_ns_21)), "crossbasis_ns_df4_ns_df3_lag0_21.csv")

# ---------------------------------------------------------------------------
# 4. Knot utilities
# ---------------------------------------------------------------------------
cat("\n=== knot utilities ===\n")

save_csv(data.frame(knots = logknots(10, nk = 3)), "logknots_lag10_nk3.csv")
save_csv(data.frame(knots = logknots(21, nk = 4)), "logknots_lag21_nk4.csv")
save_csv(data.frame(knots = equalknots(0:30, nk = 3)), "equalknots_lag30_nk3.csv")
save_csv(data.frame(knots = equalknots(0:30, nk = 4)), "equalknots_lag30_nk4.csv")

# ---------------------------------------------------------------------------
# 5. crosspred  (use cb_ns_ns and coef_12 from above)
# ---------------------------------------------------------------------------
cat("\n=== crosspred ===\n")

pred_at <- seq(-10, 35, by = 1)  # 46 values
cen_val <- 20

pred <- crosspred(
  cb_ns_ns,
  coef = coef_12,
  vcov = vcov_12,
  at   = pred_at,
  cen  = cen_val
)

save_csv(data.frame(allfit = pred$allfit),  "crosspred_allfit.csv")
save_csv(data.frame(allse  = pred$allse),   "crosspred_allse.csv")
save_csv(data.frame(alllow = pred$alllow),  "crosspred_alllow.csv")
save_csv(data.frame(allhigh = pred$allhigh),"crosspred_allhigh.csv")
save_csv(as.data.frame(pred$matfit),        "crosspred_matfit.csv")
save_csv(as.data.frame(pred$matse),         "crosspred_matse.csv")
save_csv(as.data.frame(pred$matlow),        "crosspred_matlow.csv")
save_csv(as.data.frame(pred$mathigh),       "crosspred_mathigh.csv")

# Save predvar so Python tests know the grid
save_csv(data.frame(predvar = pred_at), "crosspred_predvar.csv")

# ---------------------------------------------------------------------------
# 6. crossreduce
# ---------------------------------------------------------------------------
cat("\n=== crossreduce ===\n")

# Overall
red_overall <- crossreduce(
  cb_ns_ns,
  coef       = coef_12,
  vcov       = vcov_12,
  model.link = "identity",
  type       = "overall",
  at         = pred_at,
  cen        = cen_val
)
save_csv(data.frame(fit  = red_overall$fit),  "crossreduce_overall_fit.csv")
save_csv(data.frame(se   = red_overall$se),   "crossreduce_overall_se.csv")
save_csv(data.frame(low  = red_overall$low),  "crossreduce_overall_low.csv")
save_csv(data.frame(high = red_overall$high), "crossreduce_overall_high.csv")

# Var-specific at value=30
red_var <- crossreduce(
  cb_ns_ns,
  coef       = coef_12,
  vcov       = vcov_12,
  model.link = "identity",
  type       = "var",
  value      = 30,
  cen        = cen_val
)
save_csv(data.frame(fit  = red_var$fit),  "crossreduce_var_fit.csv")
save_csv(data.frame(se   = red_var$se),   "crossreduce_var_se.csv")
save_csv(data.frame(low  = red_var$low),  "crossreduce_var_low.csv")
save_csv(data.frame(high = red_var$high), "crossreduce_var_high.csv")

# Lag-specific at lag=5
red_lag <- crossreduce(
  cb_ns_ns,
  coef       = coef_12,
  vcov       = vcov_12,
  model.link = "identity",
  type       = "lag",
  value      = 5,
  at         = pred_at,
  cen        = cen_val
)
save_csv(data.frame(fit  = red_lag$fit),  "crossreduce_lag_fit.csv")
save_csv(data.frame(se   = red_lag$se),   "crossreduce_lag_se.csv")
save_csv(data.frame(low  = red_lag$low),  "crossreduce_lag_low.csv")
save_csv(data.frame(high = red_lag$high), "crossreduce_lag_high.csv")

cat("\nDone. All fixtures written to", FIXTURES, "\n")
