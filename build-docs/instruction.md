## Build two FastAPI endpoints as described below:


#### I. AN AGENT TO ITERATIVELY SEARCH AND FIND APPROPRIATE ACCOUNTS BASED ON A GIVEN TARGET SEGMENT OR SEED ACCOUNTS

- Input: 2 types of input:
    - Target segment: a list of keywords to tailor the search for X API
    - Seed accounts: URLs of the X accounts' profiles

- The agent is responsible to form a target segment list, writes query, searches for matched accounts, detects bot ones and returns the final list of accounts of real users. Requirements are given below:

    - If input is a list of seed accounts, it inspects and parses text from each account's bio, posts, etc., then combine them
    all into a target segment list. When extracting posts, the criteria are looking for the latest one -> earliest one
    - Query formation and automated searching based on the target segment, resulting in a list of found accounts
    - Bot account detection, read `bot-detection.md` for more details. 
    - For each account, recurrently think of its user's key information that is worth noting down, inspect and summarize them into a list such as user's bio, occupation, titles, specializations, interests, milestones, future plans, etc.  

- Output: accounts that match with input, excluding bot ones, each account accompanied by a list of its user's key information

#### II. AN AGENT THAT CHATS WITH LISTED ACCOUNTS UNDER HUMAN APPROVAL

The user selects an account in the output list of accounts to interact with.

For each account, the agent must inspect its key information, chat history, generate a message, waits for human's approval and send that message once approved. If the message is denied, receive user's feedback and propose a new improved one.
