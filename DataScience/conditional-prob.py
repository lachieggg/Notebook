# Exercise
#
# Modify the code to make it so that age has no impact on purchase
#
# Then show that the two variables are independent.

from numpy import random
random.seed(0)

totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
totalPurchases = 0
for _ in range(100000):
    ageDecade = random.choice([20, 30, 40, 50, 60, 70])
    # Probability is totally random and is not dependent
    # upon age in any meaningful way
    purchaseProbability = float(random.random())
    totals[ageDecade] += 1
    if (random.random() < purchaseProbability):
        totalPurchases += 1
        purchases[ageDecade] += 1

# Since the probability of purchasing has been modified
# and is not dependent upon age anymore, we would expect
# that Pr(E,F) = P(E) * P(F)
# E -> purchase
# F -> you are in your 30s
#

PE = float(totalPurchases)

PF = float(totals[30]) / 100000.0
print("P(30's): {}\n".format(PF))
print("P(purchase): {}\n".format(PE))

print("P(E,F) = P(E)*P(F) iff E and F are independent")

print("{} = {} * {}".format(PE*PF, PE, PF))

# 0.16646 * 6 ~= 1 which makes sense since we have 6 age groups
# Since E and F are independent, we know that Pr(B|A) == Pr(B)

for key in totals.keys():
    print(float(totals[key]) / 100000.0)

# Therefore we can see that all the age ranges have basically the
# same chance of occurring
