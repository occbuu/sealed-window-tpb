"""
Theory-driven lexicon for measuring Theory of Planned Behaviour constructs in
short-term-rental guest reviews.

The lexicon reconstructs the measurement logic published by the source article
(dictionary built from the TPB literature and validated by domain experts) and
extends it to the peer-to-peer accommodation context. Every entry is a regular
expression fragment matched case-insensitively on the lower-cased review text;
multi-word entries capture the phrasal nature of intention and norm language.

Constructs
  ATT : attitude towards the behaviour  - evaluative appraisal of the stay
  SN  : subjective norm                 - social referents, recommendation
  PBC : perceived behavioural control   - ease, convenience, controllability
  BI  : behavioural (revisit) intention - explicit forward-looking statements
  SAT : satisfaction                    - confirmation of expectations
"""

ATT = [
    r"wonderful", r"lovely", r"amazing", r"fantastic", r"excellent",
    r"delightful", r"charming", r"beautiful", r"gorgeous", r"stunning",
    r"comfortable", r"cosy", r"cozy", r"relaxing", r"peaceful", r"enjoyable",
    r"enjoyed", r"pleasant", r"great (?:stay|place|time|experience|apartment|host)",
    r"loved (?:the|it|our|this|staying)", r"we love", r"i love",
    r"worthwhile", r"worth (?:it|every|the)", r"good value",
    r"value for money", r"nicely decorated", r"tastefully", r"spotless",
    r"immaculate", r"well[- ]appointed", r"homely", r"home away from home",
    r"memorable", r"magical", r"superb", r"outstanding", r"terrific",
]

SN = [
    r"recommend", r"recommended", r"recommendation",
    r"my (?:friends?|family|wife|husband|partner|colleagues?|parents)",
    r"our (?:friends?|family|group|colleagues?)",
    r"(?:friends?|family) (?:told|said|suggested|recommended)",
    r"told (?:us|me) (?:about|to)", r"as (?:other|previous) reviews?",
    r"other (?:guests?|reviewers?|travell?ers?)",
    r"the reviews? (?:were|are|said)", r"read the reviews?",
    r"word of mouth", r"everyone (?:said|told|loved|agreed)",
    r"popular (?:with|among|choice)", r"well[- ]reviewed",
    r"highly rated", r"heard (?:about|from)", r"suggested by",
    r"would tell", r"tell (?:my|our|your) friends",
]

PBC = [
    r"easy", r"easily", r"effortless", r"convenient", r"convenience",
    r"hassle[- ]free", r"stress[- ]free", r"smooth", r"seamless",
    r"straightforward", r"simple (?:to|check|process|booking)",
    r"no (?:problem|issues?|trouble|difficulty|hassle)",
    r"self check[- ]?in", r"check[- ]?in was", r"flexible",
    r"flexibility", r"quick(?:ly)? (?:respond|replied|answered|response)",
    r"responsive", r"prompt(?:ly)?", r"always (?:available|reachable)",
    r"walking distance", r"well connected", r"close to (?:everything|the)",
    r"accessible", r"easy access", r"clear instructions",
    r"detailed instructions", r"communication was", r"communicative",
    r"got in touch", r"sorted (?:it|everything) out", r"in control",
    r"we could", r"we were able to", r"allowed us to",
]

BI = [
    r"(?:will|would|shall) (?:definitely |certainly |absolutely )?"
    r"(?:be )?(?:return|come back|stay again|book again|be back|revisit)",
    r"(?:we|i)(?:'| a)?(?:ll| will| would)? (?:be )?(?:back|returning)",
    r"come back", r"coming back", r"stay (?:here )?again", r"book again",
    r"booking again", r"next (?:time|visit|trip|stay)",
    r"hope to (?:return|come back|stay|visit)",
    r"look(?:ing)? forward to (?:return|coming|staying|our next)",
    r"can'?t wait to (?:return|come back|stay)",
    r"plan(?:ning)? to (?:return|come back|stay again)",
    r"first of many", r"definitely stay", r"our go[- ]to",
    r"would love to (?:return|come back|stay)",
]

SAT = [
    r"satisfied", r"satisfaction", r"happy with", r"very happy",
    r"pleased", r"delighted with", r"exceeded (?:our |my )?expectations?",
    r"met (?:our|my|all) expectations?", r"lived up to",
    r"as (?:described|advertised|expected|pictured)",
    r"exactly (?:as|what|like)", r"no complaints?", r"nothing (?:to|but)",
    r"perfect (?:stay|for|place|location|apartment)", r"couldn'?t (?:be|have)",
    r"thoroughly enjoyed", r"worth every (?:penny|cent|euro|dollar)",
    r"great experience", r"we were (?:very )?(?:happy|pleased|impressed)",
    r"impressed", r"first[- ]class", r"top notch", r"5 stars?", r"five stars?",
]

LEXICON = {"ATT": ATT, "SN": SN, "PBC": PBC, "BI": BI, "SAT": SAT}

# Common English function words used for a fast language screen.
EN_STOP = {
    "the", "and", "was", "were", "is", "are", "we", "our", "us", "it", "its",
    "this", "that", "there", "here", "with", "for", "very", "had", "have",
    "has", "would", "will", "not", "but", "you", "your", "they", "their",
    "from", "all", "also", "just", "his", "her", "she", "he", "been", "of",
    "in", "on", "at", "to", "a", "an", "as",
}

AUTO_PATTERNS = [
    r"this is an automated posting",
    r"the host canceled this reservation",
    r"the reservation was canceled",
]
