# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/flthy625-web/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/flthy625-web/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An entry for make.fun's **gta6-twitch-ccu-forecast-01** bounty: forecast how
many people watch Grand Theft Auto VI live on Twitch in each hour of its
release day (2026-11-19, UTC) after the first, as five quantiles per hour.

## The model

For every past launch in the case's `comparables`, the entry forms the
log-ratio of each release-day hour's viewers to that launch's own hour-1
viewers. Each comparable is weighted by similarity to the target: its unlock
pattern (rolling midnight local / single global unlock / early-access
contaminated, read from the unlock notes), the closeness of its hour-1
audience size, and the closeness of its pre-release tail (mean viewers in the
6 hours before release day, relative to hour 1). The forecast for hour h is

    q_tau(h) = p1 * exp(mu_h + z_tau * sd_h)

with mu_h and sd_h the weighted mean and standard deviation of the
log-ratios at hour h. Parameters were chosen by leave-one-out
cross-validation over the comparable launches (mean log pinball loss about
0.39, against 0.93 for a persistence forecast). With no usable comparable in
a case, the entry falls back to persistence.

## Running it

Python 3.13, standard library only, no network, deterministic:

    MAKEFUN_INPUT_DIR=<dir with case .in files> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py

`tools/` holds the cross-validation harness used to choose the model.
