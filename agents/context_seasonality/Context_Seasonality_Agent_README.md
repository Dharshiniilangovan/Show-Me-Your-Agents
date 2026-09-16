# Context & Seasonality Agent

**Version:** 1.1\
**Project:** Multi-Agent Demand Forecasting & Inventory Management\
**Phase:** Phase 1 --- Data Analysis & Foundation\
**Owner:** Member 2\
**Primary downstream consumers:** Shared Context Store, Orchestrator
Agent, Demand Forecasting Agent

------------------------------------------------------------------------

## 1. Purpose

The **Context & Seasonality Agent** captures seasonal, calendar-driven,
weather-driven, promotion-driven, and event-driven demand influences
from historical QSR demand data.

Its responsibilities are to:

-   identify weekly, monthly, quarterly, and annual seasonal patterns;
-   analyze holidays and named holiday effects;
-   analyze promotions;
-   analyze special events and named event effects;
-   incorporate weather information such as temperature and
    precipitation;
-   create model-ready contextual and seasonal features;
-   calculate historical restaurant/SKU context signals;
-   attach support counts, aggregation level, and confidence metadata to
    historical signals;
-   expose a stable machine-readable contract for the Shared Context
    Store and Orchestrator.

This agent **does not forecast future demand** and **does not claim
causal effects**. It prepares contextual knowledge and features for
downstream forecasting and reasoning components.

------------------------------------------------------------------------

## 2. Position in the Multi-Agent Architecture

``` text
Raw / Processed QSR Data
          |
          v
+------------------------------+
| Context & Seasonality Agent  |
+------------------------------+
   |           |           |
   v           v           v
Seasonality  Context     Weather
Analysis     Analysis    Analysis
   \           |          /
    \          |         /
     +---------+--------+
               |
               v
        Feature Engineering
               |
               v
        Context Signal Engine
               |
      +--------+---------+
      |                  |
      v                  v
context_features.csv  sku_context_signals.csv
      |                  |
      v                  v
Forecasting Agent    Shared Context Store
                         |
                         v
                    Orchestrator
```

The Orchestrator should treat this agent as a **context provider**, not
as the final decision-maker.

------------------------------------------------------------------------

## 3. Runtime Entry Point

From the project root:

``` powershell
python main.py
```

To inspect generated outputs:

``` powershell
python scripts/inspect_results.py
```

To run tests:

``` powershell
python -m pytest -q
```

------------------------------------------------------------------------

## 4. Input Dataset

### 4.1 Primary input

Default input:

``` text
data/raw/qsr_demand_dataset.csv
```

The validated dataset used by v1.1 contains:

  Property                                       Value
  ------------------------- --------------------------
  Rows                                       1,369,500
  Date range                  2021-01-01 to 2025-12-31
  Restaurants                                       15
  Menu items                                        50
  Missing quantity values                        5,475

### 4.2 Core entity keys

The fundamental forecasting grain is:

``` text
date + restaurant_id + menu_item_id
```

These three fields are also exposed as `forecast_keys` in
`agent_context.json`.

### 4.3 Required input columns

The feature-engineering layer requires the following columns:

  --------------------------------------------------------------------------
  Variable             Expected type     Meaning           Required
  -------------------- ----------------- ----------------- -----------------
  `date`               date/datetime     Observation date  Yes

  `day_of_week_num`    integer 0--6      Day number,       Yes
                                         Monday=0 through  
                                         Sunday=6          

  `month`              integer 1--12     Calendar month    Yes

  `avg_temp_f`         numeric           Average           Yes
                                         temperature in    
                                         Fahrenheit        

  `precip_inches`      numeric           Precipitation in  Yes
                                         inches            

  `is_holiday`         binary 0/1        Whether the date  Yes
                                         is a holiday      

  `is_special_event`   binary 0/1        Whether a special Yes
                                         event is active   

  `is_promotion`       binary 0/1        Whether a         Yes
                                         promotion is      
                                         active            
  --------------------------------------------------------------------------

For full agent execution, the dataset should also contain the
entity/target columns:

  Variable          Expected type   Meaning
  ----------------- --------------- --------------------------------------------
  `restaurant_id`   string          Restaurant identifier
  `menu_item_id`    string          Menu item/SKU identifier
  `category`        string          Product/menu category
  `quantity`        numeric         Historical observed demand / quantity sold

### 4.4 Optional descriptive context columns

These are preserved when present. If missing, v1.1 safely creates them
with `"None"`.

  Variable               Meaning                        Default when absent
  ---------------------- ------------------------------ ---------------------
  `holiday_name`         Specific holiday name          `"None"`
  `special_event_name`   Specific local/special event   `"None"`
  `precip_type`          Precipitation type             `"None"`

Examples:

``` text
holiday_name = "Independence Day"
special_event_name = "Chicago Blues Festival"
precip_type = "rain"
```

Named context is retained because different holidays/events can have
very different historical demand associations.

### 4.5 Optional `is_weekend`

If `is_weekend` is already present, it is normalized to binary. If
absent, the agent derives it as:

``` text
day_of_week_num >= 5
```

Therefore Saturday and Sunday are weekends.

------------------------------------------------------------------------

## 5. Input Validation and Defensive Handling

`add_context_features()` performs defensive preprocessing before
generating features.

### Required-column validation

If any required feature-engineering column is absent, the agent raises:

``` text
ValueError: Missing required columns: [...]
```

### Date validation

`date` is converted using `pandas.to_datetime`. Invalid dates cause a
`ValueError`.

### Binary normalization

The following fields are coerced to numeric, missing values become `0`,
values are clipped to `[0, 1]`, and the final type is `int8`:

``` text
is_holiday
is_special_event
is_promotion
```

### Precipitation handling

`precip_inches` is converted to numeric. Missing values become `0`, and
negative precipitation values are clipped to `0`.

### Missing target values

The current validated run contains 5,475 missing `quantity`
observations. Analytical aggregations drop missing `quantity` values
when calculating historical means/counts. The run summary reports the
missing-target count.

------------------------------------------------------------------------

# 6. Features Generated by the Agent

## 6.1 Calendar features

  Feature            Type      Description
  ------------------ --------- -------------------------
  `week_of_year`     integer   ISO week number
  `quarter`          integer   Calendar quarter 1--4
  `day_of_month`     integer   Day number within month
  `day_of_year`      integer   Day number within year
  `is_month_start`   binary    1 on first day of month
  `is_month_end`     binary    1 on last day of month
  `is_weekend`       binary    1 on Saturday/Sunday

------------------------------------------------------------------------

## 6.2 Cyclical seasonality features

Calendar values are cyclical. December is close to January and Sunday is
close to Monday. Sine/cosine encodings preserve this structure.

### Day-of-week

``` text
dow_sin = sin(2*pi*day_of_week_num/7)
dow_cos = cos(2*pi*day_of_week_num/7)
```

### Month

``` text
month_sin = sin(2*pi*(month-1)/12)
month_cos = cos(2*pi*(month-1)/12)
```

### Day-of-year

``` text
doy_sin = sin(2*pi*(day_of_year-1)/365.25)
doy_cos = cos(2*pi*(day_of_year-1)/365.25)
```

Generated variables:

``` text
dow_sin
dow_cos
month_sin
month_cos
doy_sin
doy_cos
```

------------------------------------------------------------------------

## 6.3 Weather features

### `has_precipitation`

``` text
1 if precip_inches > 0
0 otherwise
```

### `temp_c`

Converted from Fahrenheit:

``` text
temp_c = (avg_temp_f - 32) * 5/9
```

### `temp_band`

Temperature is also categorized into:

  Band         Fahrenheit range
  ------------ ------------------
  `freezing`   \<= 32°F
  `cold`       \>32°F to 50°F
  `mild`       \>50°F to 68°F
  `warm`       \>68°F to 86°F
  `hot`        \>86°F

`temp_band` is mainly useful for descriptive weather analysis.
Continuous temperature should normally be retained for forecasting.

------------------------------------------------------------------------

## 6.4 Context intensity features

### `context_event_count`

Number of simultaneously active major context conditions:

``` text
is_holiday + is_special_event + is_promotion
```

Possible range:

``` text
0 to 3
```

### `is_context_day`

``` text
1 if context_event_count > 0
0 otherwise
```

------------------------------------------------------------------------

## 6.5 Interaction features

These allow downstream models to learn that combined conditions may
behave differently from isolated conditions.

  Feature                   Formula
  ------------------------- -----------------------------------
  `weekend_promotion`       `is_weekend * is_promotion`
  `holiday_promotion`       `is_holiday * is_promotion`
  `event_promotion`         `is_special_event * is_promotion`
  `precipitation_weekend`   `has_precipitation * is_weekend`

Example: a Saturday promotion can behave differently from a Tuesday
promotion.

------------------------------------------------------------------------

# 7. Seasonal Analysis

The agent creates descriptive seasonal summaries by aggregating
historical `quantity`.

## 7.1 Weekday seasonality

Output:

``` text
data/outputs/seasonality_weekday.csv
```

Typical columns:

``` text
day_of_week_num
day_of_week
mean
median
count
seasonal_index
dimension
```

### `seasonal_index`

The seasonal index compares a group's mean demand with the overall
reference mean.

Interpretation:

``` text
seasonal_index = 1.00  -> average
seasonal_index > 1.00  -> above average
seasonal_index < 1.00  -> below average
```

For the validated dataset, Saturday had a seasonal index around `1.191`,
indicating historically higher demand than the overall average.

## 7.2 Monthly seasonality

Output:

``` text
data/outputs/seasonality_month.csv
```

Columns include:

``` text
month
mean
median
count
seasonal_index
dimension
```

## 7.3 Quarterly seasonality

Output:

``` text
data/outputs/seasonality_quarter.csv
```

## 7.4 Week-of-year seasonality

Output:

``` text
data/outputs/seasonality_week_of_year.csv
```

These files are descriptive summaries for analysis, explanation,
monitoring, and shared context. The forecasting model should primarily
use the row-level engineered features.

------------------------------------------------------------------------

# 8. Context Impact Analysis

The agent analyzes historical demand associations for:

``` text
promotion
holiday
special event
weather
```

These are **descriptive associations, not causal estimates**.

The generated files explicitly include:

``` text
interpretation = descriptive_association_not_causal
```

## 8.1 Historical uplift calculation

For a binary context:

``` text
uplift_pct =
    ((active_mean / baseline_mean) - 1) * 100
```

Example interpretation:

``` text
uplift_pct = +20
```

means historical mean demand during active observations was
approximately 20% higher than the comparison baseline.

It does **not** mean the context caused a 20% increase.

------------------------------------------------------------------------

## 8.2 Promotion analysis

Output:

``` text
data/outputs/promotion_impact.csv
```

Schema:

  Field              Meaning
  ------------------ ------------------------------------------
  `is_promotion`     Active context flag
  `mean`             Mean quantity under promotion
  `median`           Median quantity under promotion
  `count`            Number of valid active observations
  `baseline_mean`    Mean quantity when promotion is inactive
  `baseline_count`   Number of valid baseline observations
  `uplift_pct`       Historical descriptive uplift
  `interpretation`   Causality warning

------------------------------------------------------------------------

## 8.3 Holiday analysis

Output:

``` text
data/outputs/holiday_impact.csv
```

Schema:

``` text
is_holiday
holiday_name
mean
median
count
baseline_mean
baseline_count
uplift_pct
interpretation
```

The actual holiday identity is retained because holiday effects differ
substantially.

------------------------------------------------------------------------

## 8.4 Event analysis

Output:

``` text
data/outputs/event_impact.csv
```

Schema:

``` text
is_special_event
special_event_name
mean
median
count
baseline_mean
baseline_count
uplift_pct
interpretation
```

------------------------------------------------------------------------

# 9. Weather Analysis

## 9.1 Temperature

Output:

``` text
data/outputs/weather_temperature.csv
```

Schema:

``` text
temp_band
mean
median
count
```

## 9.2 Precipitation presence

Output:

``` text
data/outputs/weather_precipitation.csv
```

Schema:

``` text
has_precipitation
mean
median
count
```

## 9.3 Precipitation type

Output:

``` text
data/outputs/weather_precip_type.csv
```

Schema:

``` text
precip_type
mean
median
count
```

These files describe historical associations. Weather and seasonality
are correlated, so these values must not be interpreted as isolated
causal weather effects.

------------------------------------------------------------------------

# 10. Restaurant/SKU Context Signal Engine

Output:

``` text
data/outputs/sku_context_signals.csv
```

This is the primary **agent-knowledge output** for the Shared Context
Store.

Each row represents:

``` text
restaurant_id + menu_item_id
```

and includes category, baseline demand, and historical contextual
signals.

## 10.1 Baseline fields

``` text
restaurant_id
menu_item_id
category
baseline_demand
baseline_observation_count
```

`baseline_demand` is the historical mean quantity for that
restaurant/SKU combination over available non-null observations.

------------------------------------------------------------------------

## 10.2 Context signal families

The file contains signals for:

``` text
weekend
holiday
event
promotion
precipitation
```

For every signal `<context>`, the following metadata is emitted:

``` text
<context>_uplift_pct
<context>_active_count
<context>_baseline_count
<context>_signal_level
<context>_confidence
```

Example:

``` text
promotion_uplift_pct
promotion_active_count
promotion_baseline_count
promotion_signal_level
promotion_confidence
```

------------------------------------------------------------------------

# 11. Hierarchical Fallback Logic

A restaurant/SKU may not have enough observations for a reliable
granular contextual estimate.

The agent therefore evaluates context signals through this hierarchy:

``` text
restaurant + SKU
       |
       | insufficient support
       v
SKU across restaurants
       |
       | insufficient support
       v
category
       |
       | insufficient support
       v
global
```

Signal-level values are:

``` text
restaurant_sku
sku
category
global
```

### Current minimum support thresholds

``` text
MIN_ACTIVE = 10
MIN_BASELINE = 30
```

A level is selected when:

-   active count \>= 10;
-   baseline count \>= 30;
-   uplift is available.

Otherwise, the next broader level is tried.

### Important interpretation

`signal_level` describes **where the estimate came from**.

For example:

``` text
promotion_signal_level = restaurant_sku
```

means the estimate uses that restaurant and SKU.

``` text
promotion_signal_level = sku
```

means restaurant-level evidence was insufficient and the agent used that
SKU across restaurants.

------------------------------------------------------------------------

# 12. Signal Confidence

Confidence is based on:

``` text
support = min(active_count, baseline_count)
```

Current rules:

    Support Confidence
  --------- ------------
    \>= 100 `high`
     30--99 `medium`
      \< 30 `low`

Confidence describes **amount of supporting historical evidence**, not
causal certainty and not forecast accuracy.

A signal can therefore be:

``` text
signal_level = restaurant_sku
confidence = low
```

This means a granular estimate exists and passed the minimum fallback
threshold, but the amount of evidence is still limited.

The Orchestrator should preserve both fields rather than treating them
as the same concept.

------------------------------------------------------------------------

# 13. `sku_context_signals.csv` Complete Column Pattern

The current output follows this structure:

``` text
restaurant_id
menu_item_id
category
baseline_demand
baseline_observation_count

weekend_uplift_pct
weekend_active_count
weekend_baseline_count
weekend_signal_level
weekend_confidence

holiday_uplift_pct
holiday_active_count
holiday_baseline_count
holiday_signal_level
holiday_confidence

event_uplift_pct
event_active_count
event_baseline_count
event_signal_level
event_confidence

promotion_uplift_pct
promotion_active_count
promotion_baseline_count
promotion_signal_level
promotion_confidence

precipitation_uplift_pct
precipitation_active_count
precipitation_baseline_count
precipitation_signal_level
precipitation_confidence
```

------------------------------------------------------------------------

# 14. Model-Ready Feature Output

Output:

``` text
data/processed/context_features.csv
```

This is the primary row-level output for the **Demand Forecasting
Agent**.

It contains the original dataset columns plus engineered context and
seasonality variables.

A sample is also written to:

``` text
data/processed/context_features_sample.csv
```

The sample exists for inspection/debugging and should not be treated as
the full training dataset.

------------------------------------------------------------------------

# 15. Machine-Readable Agent Contract

Output:

``` text
data/outputs/agent_context.json
```

This is the recommended first file for the **Orchestrator / Shared
Context Store** to read when discovering this agent's output contract.

Current structure:

``` json
{
  "agent": "ContextSeasonalityAgent",
  "version": "1.1",
  "summary": {
    "rows": 1369500,
    "start_date": "2021-01-01",
    "end_date": "2025-12-31",
    "restaurants": 15,
    "menu_items": 50,
    "missing_quantity": 5475,
    "output_files": []
  },
  "feature_contract": {
    "forecast_keys": [
      "date",
      "restaurant_id",
      "menu_item_id"
    ],
    "target": "quantity",
    "categorical_context_features": [
      "holiday_name",
      "special_event_name",
      "precip_type"
    ],
    "numeric_context_features": [
      "dow_sin",
      "dow_cos",
      "month_sin",
      "month_cos",
      "doy_sin",
      "doy_cos",
      "week_of_year",
      "quarter",
      "day_of_month",
      "is_month_start",
      "is_month_end",
      "is_weekend",
      "is_holiday",
      "is_special_event",
      "is_promotion",
      "avg_temp_f",
      "temp_c",
      "precip_inches",
      "has_precipitation",
      "context_event_count",
      "is_context_day",
      "weekend_promotion",
      "holiday_promotion",
      "event_promotion",
      "precipitation_weekend"
    ],
    "historical_signal_file": "sku_context_signals.csv",
    "signal_note": "Historical uplifts are descriptive associations, not causal effects."
  }
}
```

`output_files` contains runtime-specific paths and should not be
hard-coded by the Orchestrator.

------------------------------------------------------------------------

# 16. Recommended Orchestrator Integration Contract

## 16.1 Agent identity

``` json
{
  "agent_name": "ContextSeasonalityAgent",
  "version": "1.1",
  "role": "context_provider"
}
```

## 16.2 What the Orchestrator can request conceptually

The current implementation is batch/file based rather than an online RPC
API. The Shared Context Store can expose queries such as:

``` text
Get context signals for restaurant R01 / menu item M08
Get promotion sensitivity for R01 / M08
Get holiday sensitivity for R01 / M08
Get event sensitivity for R01 / M08
Get precipitation sensitivity for R01 / M08
Get seasonality summary for weekday/month/quarter/week-of-year
Get forecast-ready contextual feature definitions
```

## 16.3 Recommended lookup key for historical signals

``` text
restaurant_id + menu_item_id
```

## 16.4 Recommended Orchestrator response object

When the Shared Context Store retrieves one SKU record, normalize it
into a structure similar to:

``` json
{
  "restaurant_id": "R01",
  "menu_item_id": "M08",
  "category": "Burgers",
  "baseline_demand": 19.836443,
  "signals": {
    "weekend": {
      "uplift_pct": 20.857622,
      "active_count": 520,
      "baseline_count": 1302,
      "signal_level": "restaurant_sku",
      "confidence": "high"
    },
    "holiday": {
      "uplift_pct": 18.755319,
      "active_count": 61,
      "baseline_count": 1761,
      "signal_level": "restaurant_sku",
      "confidence": "medium"
    },
    "event": {
      "uplift_pct": 78.761149,
      "active_count": 55,
      "baseline_count": 1767,
      "signal_level": "restaurant_sku",
      "confidence": "medium"
    },
    "promotion": {
      "uplift_pct": 27.133354,
      "active_count": 283,
      "baseline_count": 26997,
      "signal_level": "sku",
      "confidence": "high"
    },
    "precipitation": {
      "uplift_pct": -5.660421,
      "active_count": 628,
      "baseline_count": 1194,
      "signal_level": "restaurant_sku",
      "confidence": "high"
    }
  }
}
```

This is a recommended integration representation; the existing CSV
remains the source output.

------------------------------------------------------------------------

# 17. How the Orchestrator Should Use the Outputs

## Use `agent_context.json` for

-   agent discovery;
-   version checking;
-   feature contract discovery;
-   locating the historical signal output;
-   identifying forecasting keys and target;
-   understanding categorical vs numeric context features.

## Use `sku_context_signals.csv` for

-   explaining historical contextual sensitivity;
-   retrieving restaurant/SKU-specific context knowledge;
-   identifying whether a signal came from restaurant-SKU, SKU,
    category, or global evidence;
-   checking evidence counts and confidence;
-   supporting agent reasoning and contextual explanations.

## Use `context_features.csv` for

-   training/validation/test input to the Demand Forecasting Agent;
-   model feature selection;
-   downstream feature pipelines.

## Use seasonal/context summary CSVs for

-   dashboards;
-   diagnostics;
-   explanations;
-   monitoring;
-   exploratory reasoning.

------------------------------------------------------------------------

# 18. Important Rules for Downstream Agents

### Rule 1 --- Do not treat uplift as causal

Correct:

> Historically, demand for this SKU was 27% higher during promotion
> observations.

Incorrect:

> Running a promotion will cause demand to increase by 27%.

### Rule 2 --- Always inspect confidence

A `low` signal should be treated more cautiously than `high`.

### Rule 3 --- Always inspect signal level

`restaurant_sku` is more granular than `sku`, `category`, or `global`.

### Rule 4 --- Do not confuse confidence with forecast probability

`promotion_confidence = high` means high historical sample support. It
does **not** mean there is a high probability that future demand will
increase by the historical uplift.

### Rule 5 --- Forecasting model has final predictive responsibility

The Context & Seasonality Agent provides features and historical
context. The Demand Forecasting Agent should jointly model seasonality,
promotion, weather, event, restaurant, SKU, and other available
features.

### Rule 6 --- Named future context must be known or forecast

For future forecasting dates, variables such as future promotions,
holidays, scheduled events, and weather forecasts must be supplied by
upstream systems/data sources. Historical observed future weather must
never be leaked into model evaluation.

------------------------------------------------------------------------

# 19. Output File Inventory

  -----------------------------------------------------------------------------------------
  File                             Purpose                          Main Consumer
  -------------------------------- -------------------------------- -----------------------
  `context_features.csv`           Full model-ready context dataset Forecasting Agent

  `context_features_sample.csv`    Small inspection/debug sample    Developer

  `seasonality_weekday.csv`        Weekly seasonal summary          Shared Context /
                                                                    Dashboard

  `seasonality_month.csv`          Monthly seasonal summary         Shared Context /
                                                                    Dashboard

  `seasonality_quarter.csv`        Quarterly seasonal summary       Shared Context /
                                                                    Dashboard

  `seasonality_week_of_year.csv`   Week-of-year summary             Shared Context /
                                                                    Dashboard

  `holiday_impact.csv`             Named holiday historical         Shared Context /
                                   associations                     Orchestrator

  `promotion_impact.csv`           Overall promotion association    Shared Context /
                                                                    Orchestrator

  `event_impact.csv`               Named event historical           Shared Context /
                                   associations                     Orchestrator

  `weather_temperature.csv`        Temperature-band summary         Shared Context /
                                                                    Analysis

  `weather_precipitation.csv`      Precipitation/no-precipitation   Shared Context /
                                   summary                          Analysis

  `weather_precip_type.csv`        Precipitation-type summary       Shared Context /
                                                                    Analysis

  `sku_context_signals.csv`        Restaurant/SKU contextual        **Orchestrator / Shared
                                   knowledge + confidence           Context**

  `agent_context.json`             Machine-readable agent/feature   **Orchestrator / Shared
                                   contract                         Context**
  -----------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 20. Output Directories

``` text
data/
├── processed/
│   ├── context_features.csv
│   └── context_features_sample.csv
│
└── outputs/
    ├── seasonality_weekday.csv
    ├── seasonality_month.csv
    ├── seasonality_quarter.csv
    ├── seasonality_week_of_year.csv
    ├── holiday_impact.csv
    ├── promotion_impact.csv
    ├── event_impact.csv
    ├── weather_temperature.csv
    ├── weather_precipitation.csv
    ├── weather_precip_type.csv
    ├── sku_context_signals.csv
    └── agent_context.json
```

------------------------------------------------------------------------

# 21. Current Versioned Interface

The current stable interface is:

``` text
Agent: ContextSeasonalityAgent
Version: 1.1
```

Any future change that removes/renames output fields should increment
the agent version.

Backward-compatible additions may use a minor version increment.

The Orchestrator should check the version before assuming a schema.

------------------------------------------------------------------------

# 22. What This Agent Does NOT Do

The Context & Seasonality Agent does **not**:

-   produce the final future demand forecast;
-   optimize inventory;
-   calculate order quantities;
-   select suppliers;
-   manage expiry;
-   make procurement decisions;
-   claim causal effects from observational uplifts;
-   independently collect future weather/event/promotion data;
-   replace the Customer/Pattern Agent;
-   replace the Data Analyst Agent;
-   replace the Orchestrator.

Its boundary is:

> **Transform context + seasonality data into validated descriptive
> knowledge and forecast-ready features.**

------------------------------------------------------------------------

# 23. Dependencies on Other Agents

## Data Analyst Agent

Expected to provide clean/unified historical data and maintain the
central source of truth.

## Customer / Pattern Agent

Provides customer/product-demand relationship signals that are outside
this agent's scope.

## Shared Context Store

Stores/exposes the structured outputs generated by this agent.

## Orchestrator

Coordinates when contextual information is required and combines it with
other agent outputs.

## Demand Forecasting Agent

Consumes `context_features.csv` / equivalent shared features and
determines whether contextual variables improve predictive performance.

------------------------------------------------------------------------

# 24. Testing

Run:

``` powershell
python -m pytest -q
```

The test suite should validate at least:

-   core feature generation;
-   optional context-column fallback;
-   preservation of named holiday/event/precipitation context;
-   hierarchical signal fallback and signal metadata;
-   import/package behavior.

Do not deploy/integrate a changed version if tests fail.

------------------------------------------------------------------------

# 25. Integration Checklist for the Orchestrator Team

Before integration, confirm:

-   [ ] `agent_context.json` exists.
-   [ ] Agent version is supported.
-   [ ] `sku_context_signals.csv` exists.
-   [ ] `restaurant_id` and `menu_item_id` are available.
-   [ ] Signal level is preserved.
-   [ ] Signal confidence is preserved.
-   [ ] Active and baseline counts are preserved.
-   [ ] Historical uplift is labelled descriptive/non-causal.
-   [ ] Named holiday/event context is preserved.
-   [ ] Forecasting pipeline can access row-level context features.
-   [ ] Runtime-specific absolute file paths are not hard-coded.
-   [ ] Missing/unknown context is handled explicitly.
-   [ ] Future weather/promotions/events come from legitimate
    future-known inputs.

------------------------------------------------------------------------

# 26. Recommended Shared Context Representation

For integration, it is useful to separate **metadata**, **forecast
features**, and **historical context knowledge**.

``` text
context_seasonality/
├── metadata
│   ├── agent_name
│   ├── version
│   ├── last_run
│   └── data_period
│
├── feature_contract
│   ├── forecast_keys
│   ├── target
│   ├── numeric_features
│   └── categorical_features
│
├── seasonal_summaries
│   ├── weekday
│   ├── month
│   ├── quarter
│   └── week_of_year
│
├── context_summaries
│   ├── holidays
│   ├── promotions
│   ├── events
│   └── weather
│
└── sku_signals
    └── restaurant_id + menu_item_id
```

This keeps the Orchestrator independent of the internal implementation
details of the agent.

------------------------------------------------------------------------

# 27. Example Orchestrator Reasoning

Suppose the Orchestrator asks for context on `R01/M08`.

The Shared Context Store may return:

``` text
Weekend:       +20.86%, HIGH, restaurant-SKU
Holiday:       +18.76%, MEDIUM, restaurant-SKU
Event:         +78.76%, MEDIUM, restaurant-SKU
Promotion:     +27.13%, HIGH, SKU-level fallback
Precipitation: -5.66%, HIGH, restaurant-SKU
```

A correct downstream interpretation is:

> R01/M08 historically shows higher demand on weekends and during event
> observations. The promotion estimate uses SKU-level rather than
> restaurant-SKU evidence because the granular history was insufficient.
> These values should be treated as contextual historical signals and
> combined with the forecasting model rather than directly added
> together.

The Orchestrator should **not sum these uplifts** to produce a demand
forecast.

------------------------------------------------------------------------

# 28. Design Principle

The agent intentionally separates two outputs:

### Predictive features

``` text
context_features.csv
```

Used by ML/time-series forecasting.

### Agent knowledge

``` text
sku_context_signals.csv
```

Used by the Shared Context Store, Orchestrator, explanations, and
diagnostics.

This separation prevents descriptive historical statistics from being
incorrectly treated as the forecasting algorithm itself.

------------------------------------------------------------------------

# 29. Summary

The Context & Seasonality Agent v1.1 provides a stable Phase-1 component
that:

1.  validates contextual inputs;
2.  engineers calendar and cyclical seasonal features;
3.  preserves named holidays and special events;
4.  creates weather and context interaction variables;
5.  produces weekly/monthly/quarterly/week-of-year summaries;
6.  calculates descriptive holiday, event, promotion, and weather
    associations;
7.  generates restaurant/SKU context signals;
8.  uses hierarchical fallback for sparse signals;
9.  exposes observation counts and confidence metadata;
10. publishes a machine-readable feature contract for downstream agents.

For Orchestrator integration, the two most important files are:

``` text
data/outputs/agent_context.json
data/outputs/sku_context_signals.csv
```

For the Demand Forecasting Agent, the primary file is:

``` text
data/processed/context_features.csv
```
