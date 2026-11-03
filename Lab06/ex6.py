import arviz as az
import matplotlib
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Date
n = 10
total = 180
print("Observed rate =", total / n, "calls/hour")

# Prior
# Am ales alpha si beta astfel incat:
#   mean = alpha0/beta0 = 20  (aproape de 18)
#   sd   = sqrt(alpha0)/beta0 ~ 14
alpha0 = 2.0
beta0 = 0.1
print(f"Prior: Gamma({alpha0:g}, {beta0:g}), mean = {alpha0 / beta0:g}")

# Posterior
alpha_post = alpha0 + total
beta_post = beta0 + n
posterior = stats.gamma(a=alpha_post, scale=1.0 / beta_post)

print(f"\nPosterior: Gamma({alpha_post:g}, {beta_post:g})")
print(f"Posterior mean = {posterior.mean():.4f}")
print(f"Posterior sd = {posterior.std():.4f}")
print(f"Posterior median = {posterior.median():.4f}")

# cea mai porbabila valoare pt lambda
mode = (alpha_post - 1) / beta_post
print(f"Most probable value (mode) = {mode:.4f}")


samples = posterior.rvs(size=1_000_000, random_state=42)
hdi_az = az.hdi(samples, prob=0.94)
print(f"94% HDI (arviz, sample) = [{hdi_az[0]:.4f}, {hdi_az[1]:.4f}]")

ax = az.plot_posterior(
    {"lam": samples[:200_000]}, hdi_prob=0.94, point_estimate="mean", round_to=3
)
ax.set_title("Posterior  Gamma(182, 10.1)  for  λ")
plt.savefig("ex6_plot.png", dpi=120)
print("\nSaved ex6_plot.png")
