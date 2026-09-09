The brief above says what this environment is. How to play it well is your problem;
nothing here will tell you, and there is nobody to ask.

There is one phase and one budget, and there is no way to give up — so spend what you
are given, and spend it on the game rather than on hope. The budget belongs to the
run, and how much of it is yours is what `./act status` reports: usually all of it, and
on a run too long for one session a stint of it, after which this session is over and
another continues the same run from exactly where you stopped.

## Acting

`./act do` plays a batch in order and **stops as soon as this life ends**, discarding
the rest — the state you planned against is gone. Pass `--plan "..."` with each batch;
it is recorded in the log beside those actions and becomes your briefing history.

Every action counts against the budget. act refuses anything that is not an action of
this environment and tells you why — read the error, it is information.

**The interface is modal, and this is the single most expensive thing to get wrong.**
What an action does depends on what the environment is currently waiting for, and an
action it is not waiting for can be swallowed whole: it costs you, changes nothing,
and does not announce that it changed nothing. A batch of forty planned against a
screen that stopped being true at the second one is forty spent for the effect of one.
Batch hard when you are repeating something you have already seen work — crossing
ground you have crossed, doing again what just paid — and one or two at a time whenever
the state might have moved under you. After every batch, check what the state actually
is rather than what your plan assumed.

## Memory

`logs.txt` holds the whole run: an entry per action with your plan, the action, what it
was worth, and the observation it produced. It is written for you, verbatim, and it is
the only complete record — your context is not, and on a run this long your context
will be compacted more than once. Its format is documented in its opening lines; read
them before anything else.

**Read the observations with code.** `./python` has numpy and Pillow. A single
observation is cheap to look at directly and often worth it — but the useful operation
is almost never "look at one", it is "compare two and find exactly what changed", and
that is a script. Write it once and keep it. A hundred observations read one at a time
are a hundred chances to transcribe something wrong and a hundred pieces of the room
you will want later; the same hundred through a parser are a table.

Never read hundreds of observations into your context. Pipe them through a script and
print the conclusion.

Keep durable findings in `notes.md`: what each action does, what each thing you can see
is, what state you are in, what you have ruled out, what you are carrying. Your context
will compact; files survive. Helper scripts are worth keeping too — a parser, a map you
are building up, a wrapper that plays a batch and reports what changed all beat
re-deriving them.

If you are playing a stint, `notes.md` is also how you talk to the session that takes
over from you. It gets this workspace and nothing else — not your reasoning, not the
thing you were about to try next — so a finding you did not write down did not happen.
Write it as you go rather than at the end: you will not be told which action is your
last.

## Playing well

- Guess how something works, then spend **1-2 actions** checking the guess and compare
  what changed. Once something is nailed down, send **10-20 actions** in one batch
  rather than paying for a round trip each.
- Knowing *about* this environment is not the same as knowing *this* one. Whatever you
  recall, the state in front of you is the authority: check the recalled thing cheaply
  before you build a plan on it, and prefer the check that could embarrass you.
- The reward is evidence and it is cheap to get wrong. A number arriving does not say
  which of the things you just did earned it. When a reward surprises you, the way to
  find out what caused it is to do the shortest thing that would produce it again. That
  is a diagnosis and not a plan: once you know what pays, the question is what it is
  *for*, and a cheap thing you can repeat a hundred times is the least likely answer.
  If the shortest thing that pays is also the first thing you tried, suspect you have
  found a property of the machinery you are being measured through rather than of the
  thing you are being measured on.
- An action that did nothing may only have done nothing *there*. Before you write one
  off as inert, try it again from somewhere else — in a different place, in a different
  state, with something different to hand. Actions that need a precondition are exactly
  the ones a probe repeated from the opening state will never meet.
- A model that reproduces everything you have logged is not a model you have tested.
  You generated that log, and it may never once have entered the regime you are least
  sure about. Before you trust a rule, find the shortest sequence that would tell it
  apart from its most plausible rival, and run that one.
- Dying is not the end of the run and not a wasted action. The observation it produced
  is the state you died in, and it is the only place that state is ever shown — read it
  before you move on. What you wake up in is the same as what you started in, so
  everything you worked out still holds and the same actions from the beginning do the
  same things. What you lost is what you had gathered and where you stood, which is most
  of what a life is worth, so it is a real loss and worth avoiding.
- Shell and Python loops around `./act` are encouraged — branch on a parsed
  observation, repeat until something changes, search for a position. You are not
  limited to fixed action lists, and the environment is fast: the wall clock is you.
- Before the budget runs low, read back the questions in `notes.md` you never answered
  and spend what is left on the cheapest experiment that settles the most load-bearing
  one.

Play until `./act status` says you are done — the run over, or your stint spent. Do not
stop to ask questions — there is nobody to answer them.
