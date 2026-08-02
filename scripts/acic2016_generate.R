#!/usr/bin/env Rscript
# Generate the ACIC 2016 instances used by uplift-bench (R4 reviewer request W1:
# a third, independent continuous-outcome family).
#
# Requires the official competition package (Dorie et al., 2019):
#   curl -L -o /tmp/aciccomp.tar.gz https://github.com/vdorie/aciccomp/archive/refs/heads/master.tar.gz
#   tar -xzf /tmp/aciccomp.tar.gz -C /tmp && R CMD INSTALL /tmp/aciccomp-master/2016
#
# Setting choice (deterministic, documented): among the 75/77 parameter rows with
# heterogeneous treatment effects (te.hetero in {high, med}), we stratify by
# response-model family x heterogeneity x overlap and take the FIRST setting of
# each nonempty cell, giving settings {1,4,5,9,21,25,27,28,31}; replicates 1-2.
# Covariates are the real Collaborative Perinatal Project matrix (n=4802, p=58).
#
# Usage: Rscript scripts/acic2016_generate.R <output_dir>

suppressMessages(library(aciccomp2016))

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "data/acic2016"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

SETTINGS <- c(1, 4, 5, 9, 21, 25, 27, 28, 31)
REPLICATES <- 1:2

# Covariates once (factors kept as strings; one-hot encoding happens in the
# Python loader so that both sides stay deterministic).
x_path <- file.path(out_dir, "x.csv")
if (!file.exists(x_path)) {
  write.csv(input_2016, x_path, row.names = FALSE)
  cat("wrote", x_path, "\n")
}

for (s in SETTINGS) {
  for (r in REPLICATES) {
    f <- file.path(out_dir, sprintf("setting%02d_rep%d.csv", s, r))
    if (file.exists(f)) next
    d <- dgp_2016(input_2016, parameters_2016[s, ], random.seed = r)
    out <- data.frame(z = d$z, y = d$y, mu0 = d$mu.0, mu1 = d$mu.1, e = d$e)
    write.csv(out, f, row.names = FALSE)
    cat("wrote", f, sprintf("(treated=%.2f, mean tau=%.2f, sd tau=%.2f)\n",
                            mean(d$z), mean(d$mu.1 - d$mu.0), sd(d$mu.1 - d$mu.0)))
  }
}

# Provenance stamp.
writeLines(c(
  sprintf("generated: %s", format(Sys.time(), tz = "UTC")),
  sprintf("aciccomp2016 version: %s", as.character(packageVersion("aciccomp2016"))),
  sprintf("settings: %s", paste(SETTINGS, collapse = ",")),
  sprintf("replicates: %s", paste(REPLICATES, collapse = ","))
), file.path(out_dir, "PROVENANCE.txt"))
cat("done\n")
