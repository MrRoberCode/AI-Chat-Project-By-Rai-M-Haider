/**
 * Writing Templates Service
 * Pre-built professional templates for common writing tasks
 */

interface Template {
  name: string;
  description: string;
  structure: string;
  example: string;
  tips: string[];
}

const TEMPLATES: Record<string, Template> = {
  email: {
    name: "Professional Email",
    description: "Clear, concise professional email structure",
    structure: `Subject: [Clear, specific subject line]

Dear [Name / Team],

[Opening — State your purpose in the first sentence]

[Body — 1-3 short paragraphs with your main points]
- Paragraph 1: Context or background
- Paragraph 2: Main request, information, or update
- Paragraph 3: Next steps or call to action

[Closing — What you need from them and by when]

Best regards,
[Your Name]
[Your Title]
[Contact Info]`,
    example: `Subject: Q3 Project Update — Action Required by Friday

Dear Sarah,

I'm writing to share our Q3 project status and request your approval on two key decisions before our Friday deadline.

The development phase is 85% complete and on schedule. However, we've identified two areas requiring your input: the revised budget allocation ($15K over initial estimate due to scope changes) and the new third-party vendor contract.

Could you review the attached documents and confirm your approval by Thursday EOD? This will allow us to finalize contracts and maintain our launch timeline.

Best regards,
James`,
    tips: [
      "Keep subject lines under 50 characters",
      "State your purpose in the first sentence",
      "One email = one main topic",
      "Always include a clear call to action",
    ],
  },

  blog_post: {
    name: "Blog Post",
    description: "Engaging blog post structure optimized for readers and SEO",
    structure: `# [Compelling Headline — Include your main keyword]

## Introduction (100-150 words)
- Hook: surprising fact, question, or bold statement
- Establish why this matters to the reader
- Brief preview of what they'll learn

## Section 1: [H2 Subheading with keyword]
[300-400 words]
- Main point with evidence or example
- Supporting detail
- Transition to next section

## Section 2: [H2 Subheading]
[300-400 words]

## Section 3: [H2 Subheading]
[300-400 words]

## Key Takeaways
- Bullet point summary of main points

## Conclusion (100-150 words)
- Reinforce the main message
- Call to action (comment, share, try something)`,
    example: `# 7 Writing Habits That Will Transform Your Content in 30 Days

Most writers focus on talent. The best writers focus on systems.

After studying 50 top bloggers and interviewing professional copywriters, I found one truth: consistency beats brilliance every time. Here's how to build the habits that make great writing inevitable.

## 1. Write Before You Read
Your best creative thinking happens before you consume other people's ideas...`,
    tips: [
      "Headlines with numbers get 36% more clicks",
      "Use subheadings every 300 words",
      "Include one image per 300 words",
      "Aim for 8th grade reading level for maximum reach",
      "First and last paragraphs are most read — make them count",
    ],
  },

  cover_letter: {
    name: "Cover Letter",
    description: "Job application cover letter that gets interviews",
    structure: `[Your Name]
[Your Email] | [Phone] | [LinkedIn]
[Date]

[Hiring Manager Name]
[Company Name]
[Company Address]

Dear [Hiring Manager Name / Hiring Team],

OPENING PARAGRAPH (3-4 sentences)
- Name the specific role you're applying for
- Your most compelling qualification
- Why THIS company specifically

BODY PARAGRAPH 1 — Your Relevant Experience (4-5 sentences)
- Most relevant achievement with specific numbers
- How it directly relates to their needs
- What you learned or built

BODY PARAGRAPH 2 — Why This Company (3-4 sentences)  
- Specific reason you want THIS role at THIS company
- Show you've researched them
- Connect their mission to your goals

CLOSING (3-4 sentences)
- Express enthusiasm
- Mention attached resume
- Clear call to action

Sincerely,
[Your Name]`,
    example: `Dear Ms. Chen,

I'm excited to apply for the Senior Content Strategist role at Notion. Your mission to make tools that help people think more clearly resonates deeply — it's exactly the problem I've spent three years solving at HubSpot, where I grew our blog from 50K to 800K monthly readers.

In my current role, I led a content team of 8 writers, implemented an SEO strategy that generated $2.3M in attributed pipeline, and reduced content production time by 40% through better systems. These are the exact challenges your job description mentions.

I've been a Notion power user for four years and have written about productivity software for Fast Company. I'd love to bring that insider perspective to your content team.

I've attached my resume and portfolio. I'd welcome the chance to discuss how I can help Notion reach its next 10 million users.`,
    tips: [
      "Research the company — mention something specific",
      "Use numbers to quantify achievements",
      "Mirror the language from the job description",
      "One page maximum",
      "Customize every letter — generic letters get rejected",
    ],
  },

  essay: {
    name: "Academic Essay",
    description: "Structured academic essay with strong argumentation",
    structure: `# [Essay Title]

## Introduction (10% of total length)
- Hook: relevant quote, statistic, or scenario
- Background context
- Thesis statement: [Your clear, arguable main claim]

## Body Paragraph 1 — [First Main Point] (25% of total)
Topic sentence: [State the point]
Evidence: [Quote, statistic, or example]
Analysis: [Explain how this supports your thesis]
Transition: [Connect to next paragraph]

## Body Paragraph 2 — [Second Main Point] (25%)
[Same structure as above]

## Body Paragraph 3 — [Third Main Point] (25%)
[Same structure as above]

## Counterargument & Rebuttal (optional, 15%)
- Acknowledge the strongest opposing view
- Refute it with evidence

## Conclusion (10%)
- Restate thesis in new words
- Summarize key arguments
- Broader implication or call to action`,
    example: ``,
    tips: [
      "Your thesis should be arguable — not a fact",
      "Every paragraph needs a clear topic sentence",
      "Evidence without analysis is just a quote — explain what it means",
      "Use transition words: furthermore, however, consequently, in contrast",
      "Never introduce new arguments in the conclusion",
    ],
  },

  report: {
    name: "Business Report",
    description: "Professional business report structure",
    structure: `# [Report Title]
**Prepared by:** [Name] | **Date:** [Date] | **For:** [Audience]

---

## Executive Summary (1 page max)
[Write this last — it's a summary of everything below]
- Purpose of report
- Key findings (2-3 bullet points)
- Recommendations

## 1. Introduction
- Background and context
- Scope of the report
- Methodology used

## 2. Findings
### 2.1 [First Finding Area]
[Data, analysis, charts]

### 2.2 [Second Finding Area]

### 2.3 [Third Finding Area]

## 3. Analysis
- What the findings mean
- Patterns and insights
- Comparison to benchmarks or previous periods

## 4. Recommendations
1. [Action] — [Expected outcome] — [Timeline]
2. [Action] — [Expected outcome] — [Timeline]
3. [Action] — [Expected outcome] — [Timeline]

## 5. Conclusion
[Summary and next steps]

## Appendices
[Supporting data, charts, references]`,
    example: ``,
    tips: [
      "Executive summary should stand alone — busy executives only read this",
      "Use data visualization — tables and charts are more readable than walls of text",
      "Be specific in recommendations — vague suggestions get ignored",
      "Use active voice in recommendations: 'Implement X' not 'X should be implemented'",
    ],
  },

  proposal: {
    name: "Business Proposal",
    description: "Winning business proposal structure",
    structure: `# [Project/Service Name] — Proposal
**Submitted to:** [Client Name] | **Date:** [Date]

---

## 1. Executive Summary
- The problem we're solving
- Our proposed solution
- Why we're the right choice
- Investment and timeline overview

## 2. Understanding Your Challenge
- [Show you understand their specific problem]
- Current situation and pain points
- The cost of NOT solving this

## 3. Our Proposed Solution
- What we'll deliver
- How we'll do it (methodology)
- Timeline with milestones

## 4. Why Choose Us
- Relevant experience
- Case studies / results for similar clients
- Team credentials

## 5. Investment
| Deliverable | Timeline | Cost |
|-------------|----------|------|
| Phase 1     |          |      |
| Phase 2     |          |      |
| **Total**   |          |      |

## 6. Next Steps
1. [First step]
2. [Second step]
3. Sign agreement by [date]

---
*Questions? Contact [name] at [email]*`,
    example: ``,
    tips: [
      "Lead with their problem, not your credentials",
      "Be specific about deliverables — vague proposals lose to specific ones",
      "Include social proof — case studies beat claims",
      "Make the next step clear and easy",
      "Price last — after they see the value",
    ],
  },

  social_media: {
    name: "Social Media Post",
    description: "High-engagement social media content framework",
    structure: `LINKEDIN POST STRUCTURE:
[Hook — bold first line that stops the scroll]

[2-3 short paragraphs with your main insight]

[Personal story or example]

[Key takeaway or lesson]

[Call to action — question or invitation]

[3-5 relevant hashtags]

---
TWITTER/X THREAD STRUCTURE:
1/ [Hook tweet — the promise]
2/ [Context or problem]
3/ [Point 1]
4/ [Point 2]  
5/ [Point 3]
6/ [Takeaway]
7/ [CTA — follow, retweet, reply]`,
    example: `I turned down a $200K job offer last year.

Everyone thought I was crazy. My parents were worried. My friends questioned my judgment.

Here's what they didn't know:

The company had 40% annual turnover.
The manager had 4 people quit in 6 months.
The "equity" had a 5-year cliff with no acceleration clause.

Money isn't the only currency in a job offer.

Before you accept, ask about:
→ Why the last person left
→ Team tenure (how long people stay)
→ What success looks like in 90 days
→ Why the role is open

What other questions do you ask before accepting a job? 

#CareerAdvice #JobSearch #Leadership`,
    tips: [
      "First line = the only line people see before clicking 'more'",
      "Use line breaks — walls of text get scrolled past",
      "Specific numbers beat vague claims every time",
      "Ask a question at the end to drive comments",
      "Post consistently — algorithm rewards frequency",
    ],
  },

  press_release: {
    name: "Press Release",
    description: "Media-ready press release format",
    structure: `FOR IMMEDIATE RELEASE
[OR: EMBARGOED UNTIL: Date, Time]

# [Headline — Newsworthy, Active Voice, Present Tense]
## [Subheadline — Additional key detail]

**[CITY, Date]** — [Lead paragraph: Who, What, When, Where, Why — most important facts first]

[Second paragraph: Context, background, significance. Quote from key stakeholder.]

**"[Quote from executive/spokesperson],"** said [Name], [Title], [Company]. **"[Second sentence of quote that adds context or emotion.]"**

[Third paragraph: Supporting details, additional information, how this affects the reader]

[Fourth paragraph: Boilerplate — standard company description]

###

**Media Contact:**
[Name]
[Title]
[Email]
[Phone]`,
    example: `FOR IMMEDIATE RELEASE

# Pakistani EdTech Startup Raises $2M to Bring AI Learning to 50,000 Students
## Platform uses artificial intelligence to personalize education for underserved communities

**LAHORE, December 2024** — EduAI, a Lahore-based education technology company, today announced it has raised $2 million in seed funding to expand its AI-powered learning platform to 50,000 students across Pakistan by 2025.

The platform, which uses machine learning to adapt lesson difficulty and teaching style to each student, has already helped 8,000 students improve test scores by an average of 34% in pilot programs across Punjab.

"Every child learns differently, but our schools treat them all the same," said Ahmed Khan, CEO of EduAI. "Our technology gives every student a personalized tutor available 24/7."

###`,
    tips: [
      "Most important information goes FIRST (inverted pyramid)",
      "Write in third person — not 'we' or 'our'",
      "One strong quote per stakeholder",
      "Include specific numbers — they add credibility",
      "### signals end of release to journalists",
    ],
  },
};

export class WritingTemplatesService {
  static getTemplate = (templateType: string, context?: string): object => {
    const normalizedType = templateType.toLowerCase().replace(/\s+/g, "_");
    const template = TEMPLATES[normalizedType];

    if (!template) {
      // Return list of available templates
      return {
        error: `Template '${templateType}' not found`,
        available_templates: Object.keys(TEMPLATES).map((key) => ({
          id: key,
          name: TEMPLATES[key].name,
          description: TEMPLATES[key].description,
        })),
      };
    }

    return {
      template_type: normalizedType,
      name: template.name,
      description: template.description,
      structure: template.structure,
      example: template.example || "See structure above for guidance",
      pro_tips: template.tips,
      context_note: context
        ? `Adapt this template for: ${context}`
        : "Customize this template for your specific needs",
    };
  };

  static getAllTemplates = (): object => {
    return Object.keys(TEMPLATES).map((key) => ({
      id: key,
      name: TEMPLATES[key].name,
      description: TEMPLATES[key].description,
    }));
  };
}