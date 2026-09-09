{opening}

Every action you play is **one keystroke**, and the keys are the game's own. `./act
status` prints where you are and what is playable now; `./act board` says where the
latest observation is. `./act do <key> <key> ...` sends them in order — the game moves
only when you send one, so a key that turns out to do nothing is still a real
experiment and still costs. Any key may be repeated: `h*12`. Quote the ones your shell
would eat.

{observations}

There is one budget and it is shared. Every keystroke counts against it, including the
ones the game ignores, and nothing is held back for later. If you die the game starts
again from its beginning and the run carries on spending the same budget — what dying
cost you is the keys you had already spent. You wake up as the same thing in the same
place every time, so what you learned about either still holds.

Acting is sometimes rewarded. A number comes back with a key when it was worth
something; it is the change in a number the game itself puts on the screen. What this
run is judged on is the best single life in it — one life, taken as far as it goes —
so a total across lives is not the thing to grow.

`./python` is a Python with numpy and Pillow, for reading the observations.

The format of `logs.txt`, which keys you have and how large the budget is are
documented in its opening lines. Read them before anything else.
