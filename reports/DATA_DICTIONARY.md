# Modeling data dictionary

| Field | Meaning / transformation |
| --- | --- |
| id | Unique TMDB movie identifier; join/audit only |
| title | Movie title; display only |
| budget | Reported production budget; treated as nominal USD |
| revenue | Reported worldwide gross; regression target; never an input |
| runtime | Minutes; nonpositive values treated as missing |
| release_date | Parsed date; split and feature extraction |
| release_year | Calendar year; predictor and split rule |
| release_month | Calendar month; transformed into sine/cosine inputs |
| original_language | Language code; one-hot encoded |
| genres_list | Parsed multi-label genres; 20 binary indicators |
| primary_company | First-listed production company; selection does not establish financial importance |
| director | First crew entry with job Director; categorical encoding |
| success | 1 when revenue >= twice budget; classification target only |
| split | train / validation / test; audit only |
| log_budget | log1p budget; numeric model input |

Credits contain cast and crew JSON. Cast is parsed but not used by the current models.
Display columns and targets are excluded from FEATURES. Raw vote/popularity fields are excluded.
