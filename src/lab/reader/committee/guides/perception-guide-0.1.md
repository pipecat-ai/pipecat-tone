# Perception 0.1: the labelling guide

For the tone reader, which writes the reading of each user turn that Pipecat's Realtime Context Standard puts at the head of the user's message: the emotion the turn's words show, its two likeliest labels with how sure the reading is of each, and its sarcasm, the chance it is sarcastic. The LLM that reads it weighs it against the words, takes a reading below 0.5 as a guess, and acts on a clear one: it apologises once to a user who sounds angry, and answers what a sarcastic user means rather than what they said. The reader learns its readings from these labels. Questions: `lab.reader.committee.perception_emotion` and `lab.reader.committee.perception_sarcasm`.

## What you label

One turn a person says to an AI voice agent on a support call, with the agent's line just before it when there is one. You see the words only: no names, no voice, nothing else from the call. Some turns come from speech to text, with no capitals or punctuation and the odd misheard word: read what was meant.

You give:

1. **The emotion** the turn's words show, with an intensity.
2. **The act toward the agent**: what the turn does to the agent, with an intensity.
3. **Backchannel**: true only for a short sound of attention or agreement that does not take the turn.
4. **Sarcasm**, asked on its own: whether the turn is sarcastic, yes or no.

## The axioms

1. **One turn, from its words.** A reading is of the person's turn, from its words and the agent's line before it, never from a voice or a guess about the person.
2. **Emotion is the feeling the words show.** Neutral means the words show no clear feeling, not that the person feels none: go by what they say.
3. **Sarcasm is read together with the emotion.** A sarcastic turn means something other than, usually the opposite of, what its words say. Its emotion is the feeling it means: usually annoyance (angry), scorn for the agent or their help (contemptuous) or a weary letdown (sad), and amused for friendly teasing. It is never happy for a pleasure the words only pretend.
4. **The agent's line is context, not the turn.** Words that are sarcastic on their own stay sarcastic whatever the agent said before them: bad news, a plain update, a routine question or a kind word. The line before decides only when the words could go either way: "Oh, lovely." is sarcastic after news of another delay and sincere after news of a refund.
5. **A turn that could easily be meant is sincere.** A plain complaint is not sarcasm, nor is taking a downside in good spirit, nor thanks after help.

## Emotion

| Class | Means | Examples |
|---|---|---|
| neutral | No clear feeling: plain information, routine answers, questions and requests | "It's four eight two one three." · "Can you send it to my work address?" |
| happy | Pleasure, warmth, relief, gratitude, excitement, meant | "Oh, that's a relief, thank you." · "Brilliant, that's sorted then." |
| amused | Finds something funny: joking, laughing at themself, wry, teasing in fun | "Ha, that's my fault, I typed it wrong." |
| angry | Irritation up to rage, at someone or something, said straight or through sarcasm | "This is the third time I've called about this." · "Just put me through to a person." |
| sad | Sorrow, disappointment, regret, a weary letdown | "I was really looking forward to that trip." |
| afraid | Fear, worry, nervousness, alarm | "Is my money gone? I'm really worried." |
| surprised | Astonishment, disbelief, a start, good or bad | "Wait, it was delivered? When?" |
| disgusted | Revulsion at something physical or moral | "There was mould all over the bread. Gross." |
| contemptuous | Looking down on someone: scorn, disdain, sneering, sarcasm that mocks the agent or their help | "You can't even get an address right." |

Politeness is not a feeling of its own: thanks on the way off a routine call is neutral, or happy at low intensity when the words show real gladness. Worry about money, health or a deadline is afraid.

**Emotion intensity.** `low`: a trace of the feeling. `medium`: clearly felt. `high`: overwhelming, such as shouting, swearing at the agent, sobbing or panic. `neutral` is always `low`.

## Act toward the agent

One act: what the turn does to the agent, with an intensity (`low`, `medium`, `high`). Most turns are `none`.

- **request**: asks the agent to do something, with a softener ("Could you cancel it, please?"). A bare order is a **demand** ("Cancel it."), its force setting the intensity.
- **thanks**, **apology** ("Sorry, my mistake."), **praise** (sincere: "You've been brilliant.").
- **refusal** declines what the agent offered or asked; **dismiss** brushes the agent or the call off ("Forget it.").
- **accusation** says the agent or the company they speak for did wrong ("You lost my parcel."); **insult** demeans them ("You're useless."), and sarcasm that mocks the agent or their help is an insult at low intensity unless the words are harsh; **threat** says harm will come ("I'll take this to the papers.").
- **none**: plain information, answers, questions asking for information, complaints about the situation, and sarcasm about the situation or the person's own luck.

**Backchannel.** Short sounds of attention or agreement that do not take the turn: "mm", "uh-huh", "yeah", "right", "okay" while the agent explains. Its act is `none`. Not backchannel: "Okay, cancel it." (a demand), "Right, of course you can't." (sarcasm).

## Sarcasm

One label per turn: is it sarcastic, yes or no, read together with the agent's line before it when there is one.

**A turn is sarcastic when it means something other than, usually the opposite of, what its words say:** mock pleasure, mock thanks, mock praise, mock agreement, mock sympathy, pretending to be impressed, or irony about the situation or about oneself.

| Sarcastic | Not sarcastic |
|---|---|
| "Fantastic, so it's sitting in a box while I wait. Love that." after "Your order is packed and waiting to be collected." | "Fantastic, so it should be on its way soon. Thanks." after the same line |
| "Oh, perfect." after "The earliest visit is next Thursday." | "Oh, perfect." after "I can get someone out tomorrow morning." |
| "Wow, a note on my account. That fixes everything." after "I've added a note to your account." | "Thanks, it's good to know it's noted." after the same line |
| "Sure, because the last three emails worked so well." after "You'll get an email with a link." | "Okay, I'll keep an eye out for the email." after the same line |
| "Right, of course you can't." after "I'm sorry, I can't waive that fee." | "Okay, I understand you can't." after the same line |

### The sarcasm hard edges

1. **Sarcastic on its own words.** Mock pleasure, thanks, praise or agreement said together with the bad consequence or the complaint itself ("Great, another week without internet.") is sarcastic after any line, a plain update, a routine question and a kind word included.
2. **A short pleased answer with nothing of its own** ("Oh, great.", "Lovely.") is read with the line before: sarcastic when that line brings something the person would plainly not welcome, sincere otherwise.
3. **A plain complaint is not sarcasm:** "That's really annoying, it'll be late then." **Nor is taking a downside in good spirit:** "That's fine, a few days late is no big deal." **Nor is thanks after help.**
4. **Mock agreement and mock surprise count** when the person plainly does not believe it or is not impressed: "Sure, I'm sure it'll come this time." "Wow, you don't say."
5. **Irony about the situation or about oneself counts:** "Well, this day keeps getting better." "Nice one, me. Wrong address again."
6. **A joke that means what it says, a polite fiction and a plain insult are not sarcasm.** "Ha, my dog would've done a better job." "No trouble at all." "You're useless."
7. **Without a line before,** call it sarcastic only if the words alone show it.
8. **Label the words, not a guess about the person.**

## For the committee

Answer with the closed sets above and nothing else, and do not explain. If a turn is ambiguous, label the likelier reading, at a lower intensity: the reader learns how sure to be from how often the committee agrees.

Sarcasm is asked as its own short question: you get the turn and the agent's line before it, and answer yes or no.
