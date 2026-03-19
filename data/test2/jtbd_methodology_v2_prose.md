# Jobs-to-be-Done (JTBD) Methodology V2

## Overview

Jobs-to-be-Done (JTBD) is a framework for understanding customer motivation and needs through job statements. JTBD explores what job users are trying to do, what triggers them, what alternatives they consider, and what obstacles they face.

**Methodology Version**: 2.1
**Goal**: Understand what jobs users are trying to accomplish and what motivates them
**Opening Bias**: Ask about a specific recent time they made a decision or change related to the topic.

---

## Node Types: The 8 JTBD Concepts

### Job Statement (Level 0)
A core functional or emotional job a customer is trying to accomplish.

**Examples**:
- "get my daily caffeine fix"
- "feel productive in the morning"
- "socialize with colleagues"

### Job Context (Level 0)
The situation or circumstances when the job arises.

**Examples**:
- "when I'm rushing to work"
- "when I'm working from home"
- "when meeting clients"

### Job Trigger (Level 0)
An event or stimulus that initiates the job. This is what prompts the person to seek a solution.

**Examples**:
- "feeling tired in the morning"
- "anticipating a long meeting"
- "hosting a gathering"

### Pain Point (Level 0)
An obstacle or frustration in getting the job done.

**Examples**:
- "coffee is too bitter"
- "takes too long to prepare"
- "upsets my stomach"

### Gain Point (Level 0)
A desired outcome or benefit that makes the job worthwhile.

**Examples**:
- "smooth, enjoyable taste"
- "quick and convenient"
- "feeling energized without jitters"

### Solution Approach (Level 0)
A current or potential method for accomplishing the job.

**Examples**:
- "using oat milk in coffee"
- "buying coffee at a cafe"
- "meal prepping"

### Emotional Job (Level 1, Terminal)
The emotional state or identity the customer seeks to achieve.

**Examples**:
- "feel healthy and conscientious"
- "project a professional image"
- "treat myself with care"

### Social Job (Level 1, Terminal)
The social impression or relationship the customer wants to influence.

**Examples**:
- "fit in with health-conscious peers"
- "signal environmental values"
- "avoid social awkwardness"

---

## Edge Types: Relationships Between Concepts

### Occurs In
Temporal or situational relationship between job and context.

### Triggers
Causal relationship where source initiates or causes target (A triggers B means A is the cause, B is the effect).

### Addresses
Solution approach resolves a pain point or enables a gain point.

### Conflicts With
Something that prevents or undermines something else.

### Enables
Something that makes something else more achievable or satisfying.

### Supports
Something that contributes to or reinforces something else.

### Revises
Contradiction — newer belief or approach supersedes older one.

---

## Extraction Guidelines

When analyzing customer interviews, follow these principles:

1. **Listen for phrases indicating what the user is trying to accomplish** ("I need to...", "I want to...", "The goal is...")

2. **Identify the circumstances when the job becomes important** ("when...", "especially when...", "in situations like...")

3. **Extract pain points directly expressed** ("frustrating that...", "hate it when...", "problem is...")

4. **Extract gain points as positive outcomes** ("love that...", "great because...", "important that...")

5. **Distinguish between functional, emotional, and social jobs** — often all three coexist

6. **Map the current solution approach even if not explicitly stated**

7. **Connect jobs to their underlying emotional and social drivers**

8. **Look for trade-offs users make between competing jobs**

9. **Actively extract emotional jobs**: look for identity statements, feeling words, aspirational language

10. **Actively extract social jobs**: look for peer references, social signals, reputation concerns

---

## Relationship Examples

### Explicit Job Statement
**Description**: Direct statement of what needs to be accomplished
**Example**: "I need something that wakes me up but doesn't make me jittery"
**Extraction**: job_statement('wake up without jitters') → addresses → pain_point('caffeine jitters')

### Implicit Context
**Description**: Context revealed through situation description
**Example**: "When I have 8am meetings, I can't wait in line at the cafe"
**Extraction**: job_statement('get morning coffee') → occurs_in → job_context('early morning meetings')

### Pain Trigger Chain
**Description**: Trigger reveals the pain point and the job together
**Example**: "Dairy makes me feel bloated, especially before presentations"
**Extraction**: job_trigger('before presentations') → triggers → job_statement('feel comfortable') conflicts_with pain_point('dairy bloating')

### Functional Supports Emotional
**Description**: Functional choice enables emotional outcome
**Example**: "I choose oat milk because it aligns with my values"
**Extraction**: solution_approach('oat milk') → supports → emotional_job('feel environmentally responsible')

---

## Extractability Criteria

### What IS Extractable (contains)
- Goals or objectives the user wants to achieve
- Frustrations, pain points, or obstacles
- Positive outcomes or benefits they value
- Situational context (when, where, with whom)
- Current solutions or workarounds
- Emotions or social considerations related to choices

### What is NOT Extractable (contains)
- Simple yes/no responses
- Acknowledgments ('okay', 'I see', 'right')
- Questions back to the interviewer
- Purely factual product descriptions without user context
- Irrelevant tangents

---

## Concept Naming Convention

Name each concept according to its node type role:
- **phrase job_statements** as jobs to be done
- **pain_points** as frustrations or obstacles
- **gain_points** as desired outcomes or benefits
- **solution_approaches** as methods or actions
- **job_contexts** as situations or circumstances
- **emotional_jobs** as emotional states sought
- **social_jobs** as social outcomes desired
- **job_triggers** as initiating events

Use the examples in each node type description as naming models.

---

## Key Distinctions for Classification

### Pain Point vs. Gain Point
- **Pain Point**: Negative, frustrating, obstacle ("it's too slow", "I hate that...")
- **Gain Point**: Positive, desired, beneficial ("I love that...", "great when...")

### Job Trigger vs. Job Context
- **Job Trigger**: Event that STARTS the job ("when X happened, I needed to...")
- **Job Context**: Ongoing situation WHERE the job happens ("when I'm at work...", "on weekends...")

### Solution Approach vs. Job Statement
- **Job Statement**: WHAT needs to be accomplished ("I need to...")
- **Solution Approach**: HOW they're doing it ("I use...", "I tried...", "I bought...")

### Emotional Job vs. Social Job
- **Emotional Job**: How the person wants to FEEL ("I want to feel...", "It makes me feel...")
- **Social Job**: How they want to APPEAR to others ("I want to seem...", "Others think...")

---

## Distinguishing Similar Concepts in Practice

When classifying interview utterances, consider these nuances:

1. **Job Trigger vs. Job Context**: If the utterance describes a specific event that prompted action → job_trigger. If it describes ongoing circumstances → job_context.

2. **Solution Approach vs. Job Statement**: If it describes a method, tool, or action taken → solution_approach. If it describes a goal or desired outcome → job_statement.

3. **Pain Point vs. Emotional Job**: If it describes an external obstacle → pain_point. If it describes an internal emotional state → emotional_job.

4. **Functional vs. Emotional vs. Social Jobs**: The same utterance may contain all three levels. Extract all that apply.

5. **Implicit Solutions**: Even if not explicitly stated, infer the current approach from what the person describes doing regularly.

---

This methodology enables systematic extraction of customer motivations, obstacles, and solutions from interview data, creating a comprehensive understanding of the jobs customers are trying to accomplish.
