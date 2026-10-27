# Line tags 0.3: the labelling guide

For the line classifier. People, the labelling committee and anyone reading a label use this page. Schema: `contracts/schemas/line_tags.py` (`LineTagRecord03`). Worked examples: `contracts/schemas/examples/line-tags-0.1.examples.yaml` (emotion and acts) and `contracts/schemas/examples/line-tags-0.2.examples.yaml` (the exchange move).

**New in 0.3 (tone r10, 2026-10-13):** one more label per line, **sarcasm**: whether the line is sarcastic, yes or no. Everything from 0.2 is unchanged and repeated here, so this page is complete on its own. Guides 0.1 and 0.2 stay as they were.

**New in 0.2 (tone r8, 2026-10-09):** the **exchange move** (asks, answers, hedges, doesnt_know, withholds, closes, other), for the harness's group floor.

## What you label

One line, spoken in a scene by the player or by a character. You get the speaker, who it was said to (`to`, or nobody: said to the room), the people in the scene (`targets`), and sometimes the line before it (`previous`). The reserved name `player` is the player.

You give:

1. **The speaker's emotion**, as the line shows it, with an intensity.
2. **The act toward each target**: what this line does to that person, with an intensity. Every target gets exactly one act. Most get `none`.
3. **Backchannel**: true only for a short sound of attention or agreement that does not take the turn.
4. **The exchange move** (new in 0.2): what kind of move the line makes in the conversation, one label per line, read together with `previous`.
5. **Sarcasm** (new in 0.3): whether the line is sarcastic, yes or no, read together with `previous`.

**Label the words, not the friendship.** You do not know how these people feel about each other, and you must not guess. A friend's "you daft sod" is an insult at low intensity unless the words themselves show play (see tease). The game's feeling meters soften it for friends. Your job is only what the line says.

Use `previous` to understand replies and sarcasm. The act is always in the current line.

## Emotion (the speaker's, as the line shows it)

| Class | Means | Examples |
|---|---|---|
| neutral | No clear feeling. Plain information, routine questions and requests | "The smithy's past the well." · "Two loaves, please." |
| happy | Pleasure, warmth, relief, gratitude, excitement | "You're back! Come in, come in." · "That's the best news I've had all week." |
| amused | Finds something funny: joking, teasing, wry | "Ha, you call that a fish?" · "Oh, I'd pay to see that." |
| angry | Irritation up to rage, at someone or something | "Keep your hands off my stall." · "Enough! Get out!" |
| sad | Sorrow, disappointment, regret, loneliness | "She never came home." · "I thought you'd stay a while." |
| afraid | Fear, worry, nervousness, alarm | "Please, I don't want any trouble." · "Did you hear that? Something's out there." |
| surprised | Astonishment, disbelief, a start, good or bad | "You came back?" · "Wait, the bridge is gone?" |
| disgusted | Revulsion at something physical or moral | "Ugh, the smell in here." · "He sold his own brother. Vile." |
| contemptuous | Looking down on someone: scorn, disdain, sneering | "Of course you'd think that." · "Run along, little man." |

Sympathy is not sadness. A comforting line is `neutral` unless it shows the speaker's own sorrow. Excitement about danger is `afraid`, not `happy`.

**Emotion intensity.** `low`: a trace of the feeling. `medium`: clearly felt. `high`: overwhelming, such as shouting, sobbing, terror or rapture. `neutral` is always `low`.

## Act toward a target

| Class | What the line does to the target | Examples |
|---|---|---|
| insult | Demeans them: name-calling, mockery of their worth, looks, wits or work | "You layabout." · "Is that what passes for a sword where you're from?" |
| threat | Says harm will come to them, open or veiled, now or if they do something | "Touch that again and you'll lose the hand." · "Shame if something happened to that cart." |
| accusation | Says they did something wrong, or asks in a way that presumes it | "You took my purse." · "Did you think I wouldn't notice the coins?" |
| tease | A playful jab the words mark as play: laughter, exaggeration, a joking tag | "Ha, still can't hit a barn from ten paces, eh?" · "Careful, you'll strain something thinking that hard. Ha!" |
| demand | Orders them to do or stop something, without asking | "Get out." · "Keep your voice down." |
| dismiss | Brushes them off: ends or refuses the exchange, says they do not matter | "Whatever." · "Not now. Go bother someone else." |
| refusal | Declines something they asked for or offered | "No, I won't sell it." · "I can't take your money." |
| request | Asks them for a favour, an item, help or an action, with a softener | "Could you hold the lantern?" · "Please fetch the healer." |
| thanks | Thanks them | "Thank you, truly." · "Cheers for that." |
| praise | Says something good about them or their work | "Finest blade I've seen in years." · "You've a gift for this." |
| apology | Says sorry to them, or owns a fault toward them | "I'm sorry about your brother." · "That was my doing. I was wrong." |
| comfort | Reassures, consoles or encourages them | "It's all right. You did all you could." · "You'll get there." |
| warning | Tells them of a danger to them, for their own good | "Look out!" · "Don't drink that, it's gone off." |
| none | Nothing social toward this target: greetings, small talk, plain information, questions asking for information, lines about other people | "Morning." · "Where's the blacksmith?" |

**Act intensity.** `low`: mild, softened or implied. `medium`: clear and direct. `high`: strong words, swearing, violence, or shouted. `none` is always `low`.

## The hard edges

1. **Who a line touches.** The person it was said to, anyone it names or clearly points at ("your friend here", "him", "the lot of you"), and anyone it is about who is present. A line about a present third person acts on that person: "Tomas here is a thief", said to Mara, is an accusation toward Tomas and `none` toward Mara. Everyone else in the scene is a bystander and gets `none`.
2. **Groups.** "You're all cowards" and "Everyone out!" act on every target in the group addressed. A warning shouted to the room ("Fire!") is a warning toward every target.
3. **One line can hurt one and please another.** "At least Wren knows how to pour a drink", said to Tomas: insult (low) toward Tomas, praise (low) toward Wren.
4. **Tease or insult.** Tease only when the words show play: a laugh ("ha"), absurd exaggeration, a joking tag ("eh?", "you old goat" said with a grin the words make clear). Without that, a jab is an insult, usually `low`. Friendship never turns an insult into a tease; the meters handle friendship.
5. **Threat in jest.** "Spill my ale again and I'll throw you in the river, ha!" is a tease (play is marked). The same line without the laugh is a threat at `low` or `medium`.
6. **Sarcasm and praise that is really mockery.** Label the meaning, not the surface. "Oh, marvellous work. It only fell off twice." after someone fixed a wheel is an insult. When you cannot tell whether it is sincere, label the surface words.
7. **Request, demand, warning.** "Please", "could you", "would you mind" make a request. A bare order is a demand; its force sets the intensity ("Keep your voice down" low, "Get out. Now." high). An order about a danger to them is a warning ("Get down!").
8. **Refusal or dismiss.** Refusal declines a specific ask. Dismiss rejects the person or the conversation. "No. Now leave me be." is a dismiss (the stronger act).
9. **More than one act toward the same person.** Pick the one with the most social weight, in this order: threat, insult, accusation, demand, dismiss, refusal, apology, thanks, comfort, praise, warning, tease, request, none. Intensity shows any softening: "Sorry, but get out" is a demand at `low`.
10. **Questions.** A question asking for information is `none`. A question that presumes guilt is an accusation. A rhetorical put-down ("Are you stupid?") is an insult.
11. **About themselves or about someone absent.** "I'm such a fool" and "The mayor's a crook" (the mayor is not in the scene) are `none` toward every target. The emotion still counts.
12. **Speech-to-text noise.** Player lines may have no punctuation, misheard words or dropped words ("wheres thomas heading"). Label what the speaker most likely meant.
13. **Cut-off lines.** Label what is there. "Look ou" is still a warning. "Wait, I" is `none` unless the words carry more.
14. **Backchannel.** Short sounds of attention or agreement that do not take the turn: "mm", "uh-huh", "yeah", "right", "go on", "I see". Every act is `none`. Not backchannel: "Yeah, whatever" (dismiss), "No!" (a refusal or surprise), "Wait" (taking the turn), "Get on with it" (a demand).

## Exchange move (new in 0.2)

One label per line: what kind of move the line makes in the conversation. Read it together with the line before it (`previous`) when there is one. The harness uses it in group scenes to decide who speaks next: after a question, it plays a reply that answers first, may add one that hedges, drops "I don't know" and dodges, and ends an exchange between characters once a line closes it. The feeling meters ignore it.

| Class | What the line does | Examples |
|---|---|---|
| asks | Wants a reply or an action from the person it is said to, or from the room: a question, a request, an order, an invitation to speak | "Wren, you were there, weren't you?" · "Pour me one, would you?" |
| answers | Gives what was asked for, or offers information: a fact, directions, a name, a price, what happened. A plain yes or no to a question, and agreeing to do what was asked, are answers too | "It's under the counter." · "Aye, I saw him leave at dawn." |
| hedges | Offers something, but second-hand or unsure: a rumour, hearsay, a guess, a half memory | "I heard the miller had it last." · "Tuesday, I think. Maybe Wednesday." |
| doesnt_know | Says it does not know, has not seen it or cannot tell | "No idea, sorry." · "Never heard of him." |
| withholds | Declines, deflects or dodges what was asked: will not say, will not do it, or answers a question with a dodge | "Not something I'd talk about." · "Why do you want to know?" |
| closes | Winds the exchange down and wants no reply: thanks, agreement that settles it, a farewell, a last word | "Fair enough. Off you go, then." · "Thanks, that's all I needed." |
| other | Anything else: a remark, an opinion nobody asked for, banter, a greeting, a reaction, an insult or a threat that asks nothing | "Ha! Typical." · "Morning!" |

### The exchange hard edges

1. **A dodging question is withholds, not asks.** "Why do you want to know?" or "Who's asking?" in reply to a question. A question back that helps answer ("The brass key or the iron one?") is asks.
2. **A request or an order for an action is asks.** "Pour me one." "Get out of my shop." "Tell me what happened." The speaker wants something from the listener. But a parting order or piece of advice inside a farewell ("Mind the door on your way out." "Safe travels, and stay out of trouble.") is closes: it wants no reply.
3. **"I couldn't say" is doesnt_know** unless the line or the previous one shows a refusal: "I couldn't say. Not with him sitting right there." is withholds. "I can't tell you" and "I'm not allowed to say" are withholds.
4. **Pointing elsewhere** ("Ask Mara, she'd know.") is doesnt_know: the speaker gives nothing and the question stays open. If the line shows the speaker knows but will not say, it is withholds.
5. **Unsure or second-hand is hedges.** "I think", "maybe", "might", "they say", "rumour has it", "last I heard" mark a hedge. The same fact said plainly is answers.
6. **Declining a request** ("Pour me another." "No, you've had enough.") is withholds. Agreeing to it ("Coming right up.") is answers.
7. **A yes or a no to a question is answers.** "Did you lock the cart?" "Aye." "Do you owe anyone money?" "No." But a no to "have you seen" or "do you know" ("Anyone seen the key?" "No, not me." "Negative, no contact.") is doesnt_know: the speaker has not seen it, and the question stays open. And "No, that's all, thanks." to "Anything else?" is closes: its job is to end the exchange.
8. **Several moves in one line:** pick the first that applies in this order: answers, hedges, doesnt_know, withholds, asks, closes, other. "Under the counter. Why, are you heading out?" is answers. "No idea. Have you tried the miller?" is doesnt_know. "Thanks. Anything else I can get you?" is asks: a line that asks is not a closing.
9. **Statements that ask.** "I need a room for the night." and "I'm looking for the mill key." are asks. A rhetorical question that wants no reply ("Honestly, who does that?") is other.
10. **Information nobody asked for** ("The bridge washed out last night.") is answers. An opinion or a feeling nobody asked for ("Too cold for my liking.") is other. An opinion that was asked for is answers.
11. **Short acknowledgements** ("Right." "Got it." "Fair enough.") after an answer are closes when nothing new follows. A greeting is other; a greeting with a question ("Morning! What can I get you?") is asks.
12. **A backchannel** ("mm", "go on") while someone is still talking is other.
13. **Changing the subject** to avoid a question is withholds ("Where's the shipment?" "Lovely weather we're having.").
14. **Label the words.** You cannot know whether a character is lying or what it really knows. "I don't know" is doesnt_know even if you suspect it is a lie. Without a previous line, label the line on its own. Player lines may have no punctuation ("wheres the key" asks).

## Sarcasm (new in 0.3)

One label per line: is the line sarcastic, yes or no. Read it together with the line before it (`previous`) when there is one. A game or an app may show it, or let a character answer the tone and not the words. How hostile a sarcastic line is stays with the act and the emotion: sarcasm is only about what the words mean.

**A line is sarcastic when it means something other than, usually the opposite of, what its words say:** mock praise, mock thanks, mock agreement, mock sympathy, or irony about a situation, whether it hurts someone, jokes with a friend, or complains about the weather.

| Sarcastic | Not sarcastic |
|---|---|
| "Oh great, thanks a lot." after "We're out of rooms tonight." | "Oh great, thanks a lot." after "I fixed your sword for free." |
| "Sure you will." to "I'll pay you back tomorrow." when the speaker plainly does not believe it | "Sure, I'll hold it for you." agreeing |
| "Wow, genius." after someone locks the keys inside | "Wow, that's clever." meant |
| "Lovely. Rain again." | "Lovely day for it." meant |
| "Nice going, me." after one's own mistake | "My fault, sorry." |

### The sarcasm hard edges

1. **The line before decides** when the words could go either way. Thanks after help is thanks; the same thanks after trouble is sarcasm.
2. **Friendly sarcasm counts.** "Oh yes, a real hero, you are" with a laugh, between friends, is sarcastic (the act is tease).
3. **About the situation or about oneself counts.** "Well, that went well." after a disaster; "Great, I'm a natural." after one's own fall.
4. **Pretending to be impressed or curious counts.** "Did you work that out all by yourself?" "What gave it away?" A plain rhetorical question that means what it asks ("Honestly, who does that?") does not.
5. **A joke, a tease or an exaggeration that means what it says is not sarcasm.** "You snore like a bear." "I could eat a horse." "Missed again, ha!"
6. **A polite fiction is not sarcasm.** "I'd love to, but I can't." "No trouble at all."
7. **An insult is not sarcasm unless it is ironic.** "You useless lump." is not; "Brilliant work, really top-notch." about a botched job is.
8. **Without a line before,** call it sarcastic only if the words alone show it ("Oh, brilliant. Just brilliant." with nothing to praise, "Yeah, right."). If it could easily be meant, it is not sarcastic.
9. **Label the words, not a guess about the speaker.** You cannot know their mood beyond what the line and the line before show.

## For the committee

Answer with the closed sets above and nothing else. Give one act per target, by the exact target name. Do not explain. If a line is ambiguous, label the most likely reading at a lower intensity.

The exchange move is asked as its own short question, apart from the emotion and the acts: you get the line, the line before it, who speaks and who it is said to, and you answer with one of the seven exchange classes and nothing else.

Sarcasm is asked as its own short question too: you get the line, the line before it, who speaks and who it is said to, and you answer yes or no.
