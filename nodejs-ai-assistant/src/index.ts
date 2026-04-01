import cors from "cors";
import "dotenv/config";
import express from "express";
import { createAgent } from "./agents/createAgent";
import { AgentPlatform, AIAgent, ChatSummaryRequest, DocumentAnalysisRequest } from "./agents/types";
// import { documentAnalysisService } from "./services/DocumentAnalysisService";
import { documentAnalysisService } from "./agents/services/DocumentAnalysisService";


// import { WritingTemplatesService } from "./services/WritingTemplatesService";
import { WritingTemplatesService } from "./agents/services/WritingTemplatesService";
import { apiKey, serverClient } from "./serverClient";
import { GoogleGenAI } from "@google/genai";

const app = express();

// Increase payload limit for document uploads (base64 PDFs can be large)
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true, limit: "20mb" }));
app.use(cors({ origin: "*" }));

// ─── Agent Cache ───────────────────────────────────────────────────────────────
const aiAgentCache = new Map<string, AIAgent>();
const pendingAiAgents = new Set<string>();
const inactivityThreshold = 480 * 60 * 1000; // 8 hours

setInterval(async () => {
  const now = Date.now();
  for (const [userId, aiAgent] of aiAgentCache) {
    if (now - aiAgent.getLastInteraction() > inactivityThreshold) {
      console.log(`[Cleanup] Disposing inactive agent: ${userId}`);
      await disposeAiAgent(aiAgent);
      aiAgentCache.delete(userId);
    }
  }
}, 5000);

// ─── Health Check ──────────────────────────────────────────────────────────────
app.get("/", (req, res) => {
  res.json({
    message: "AI Writing Assistant Server",
    version: "2.0.0",
    provider: "Google Gemini 2.0 Flash",
    features: [
      "streaming_responses",
      "google_search_grounding",
      "duckduckgo_search",
      "document_analysis",
      "writing_templates",
      "chat_summary",
      "writing_analysis",
    ],
    activeAgents: aiAgentCache.size,
    timestamp: new Date().toISOString(),
  });
});

// ─── Agent Management ──────────────────────────────────────────────────────────
app.post("/start-ai-agent", async (req, res) => {
  const { channel_id, channel_type = "messaging" } = req.body;
  console.log(`[API] /start-ai-agent → channel: ${channel_id}`);

  if (!channel_id) {
    res.status(400).json({ error: "Missing channel_id" });
    return;
  }

  const user_id = `ai-bot-${channel_id.replace(/[!]/g, "")}`;

  try {
    if (!aiAgentCache.has(user_id) && !pendingAiAgents.has(user_id)) {
      pendingAiAgents.add(user_id);

      await serverClient.upsertUser({
        id: user_id,
        name: "AI Writing Assistant",
        image: "https://api.dicebear.com/9.x/bottts/svg?seed=gemini",
      });

      const channel = serverClient.channel(channel_type, channel_id);
      await channel.addMembers([user_id]);

      const agent = await createAgent(
        user_id,
        AgentPlatform.GEMINI,
        channel_type,
        channel_id
      );

      await agent.init();

      if (aiAgentCache.has(user_id)) {
        await agent.dispose();
      } else {
        aiAgentCache.set(user_id, agent);
      }
    } else {
      console.log(`[API] Agent ${user_id} already running or pending`);
    }

    res.json({ message: "AI Agent started", provider: "gemini" });
  } catch (error) {
    const errorMessage = (error as Error).message;
    console.error("[API] Failed to start agent:", errorMessage);
    res.status(500).json({ error: "Failed to start AI Agent", reason: errorMessage });
  } finally {
    pendingAiAgents.delete(user_id);
  }
});

app.post("/stop-ai-agent", async (req, res) => {
  const { channel_id } = req.body;
  const user_id = `ai-bot-${channel_id.replace(/[!]/g, "")}`;

  try {
    const aiAgent = aiAgentCache.get(user_id);
    if (aiAgent) {
      await disposeAiAgent(aiAgent);
      aiAgentCache.delete(user_id);
    }
    res.json({ message: "AI Agent stopped" });
  } catch (error) {
    res.status(500).json({ error: "Failed to stop AI Agent", reason: (error as Error).message });
  }
});

app.get("/agent-status", (req, res) => {
  const { channel_id } = req.query;
  if (!channel_id || typeof channel_id !== "string") {
    return res.status(400).json({ error: "Missing channel_id" });
  }
  const user_id = `ai-bot-${channel_id.replace(/[!]/g, "")}`;

  if (aiAgentCache.has(user_id)) {
    res.json({ status: "connected", provider: "gemini" });
  } else if (pendingAiAgents.has(user_id)) {
    res.json({ status: "connecting", provider: "gemini" });
  } else {
    res.json({ status: "disconnected" });
  }
});

// ─── Token Provider ────────────────────────────────────────────────────────────
app.post("/token", async (req, res) => {
  try {
    const { userId } = req.body;
    if (!userId) return res.status(400).json({ error: "userId is required" });

    const issuedAt = Math.floor(Date.now() / 1000);
    const expiration = issuedAt + 60 * 60; // 1 hour
    const token = serverClient.createToken(userId, expiration, issuedAt);

    res.json({ token });
  } catch (error) {
    res.status(500).json({ error: "Failed to generate token" });
  }
});

// ─── Document Analysis ─────────────────────────────────────────────────────────
/**
 * POST /analyze-document
 * Analyzes uploaded documents (PDF, TXT, MD) using Gemini Vision
 * Body: { base64Data, mimeType, fileName, prompt, channelId, channelType? }
 */
app.post("/analyze-document", async (req, res) => {
  const { base64Data, mimeType, fileName, prompt, channelId, channelType = "messaging" } =
    req.body as DocumentAnalysisRequest;

  if (!base64Data || !mimeType || !channelId) {
    return res.status(400).json({
      error: "Missing required fields: base64Data, mimeType, channelId",
    });
  }

  console.log(`[API] /analyze-document → ${fileName} (${mimeType})`);

  try {
    const analysis = await documentAnalysisService.analyzeDocument(
      base64Data,
      mimeType,
      fileName || "document",
      prompt || "Please analyze this document and provide a comprehensive summary with key insights."
    );

    // Optionally send result as a message in the channel
    if (channelId) {
      const channel = serverClient.channel(channelType, channelId);
      await channel.sendMessage({
        text: `📄 **Document Analysis: ${fileName || "Uploaded Document"}**\n\n${analysis}`,
        ai_generated: true,
      });
    }

    res.json({
      success: true,
      fileName,
      analysis,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("[API] Document analysis failed:", error);
    res.status(500).json({
      error: "Document analysis failed",
      reason: (error as Error).message,
    });
  }
});

/**
 * POST /summarize-document
 * Summarize a document in brief, detailed, or bullet point format
 * Body: { base64Data, mimeType, summaryLength?, channelId }
 */
app.post("/summarize-document", async (req, res) => {
  const { base64Data, mimeType, summaryLength = "detailed", channelId, channelType = "messaging" } =
    req.body;

  if (!base64Data || !mimeType) {
    return res.status(400).json({ error: "Missing base64Data or mimeType" });
  }

  try {
    const summary = await documentAnalysisService.summarizeDocument(
      base64Data,
      mimeType,
      summaryLength
    );

    if (channelId) {
      const channel = serverClient.channel(channelType, channelId);
      await channel.sendMessage({
        text: `📋 **Document Summary**\n\n${summary}`,
        ai_generated: true,
      });
    }

    res.json({ success: true, summary, summaryLength });
  } catch (error) {
    res.status(500).json({
      error: "Summary failed",
      reason: (error as Error).message,
    });
  }
});

// ─── Writing Templates ─────────────────────────────────────────────────────────
/**
 * GET /templates
 * Returns all available writing templates
 */
app.get("/templates", (req, res) => {
  const templates = WritingTemplatesService.getAllTemplates();
  res.json({ templates });
});

/**
 * GET /templates/:type
 * Returns a specific writing template
 */
app.get("/templates/:type", (req, res) => {
  const { type } = req.params;
  const { context } = req.query;

  const template = WritingTemplatesService.getTemplate(
    type,
    context as string | undefined
  );

  res.json({ template });
});

/**
 * POST /templates/apply
 * Apply a template and send it to a channel as an AI message
 */
app.post("/templates/apply", async (req, res) => {
  const { templateType, context, channelId, channelType = "messaging" } = req.body;

  if (!templateType || !channelId) {
    return res.status(400).json({ error: "Missing templateType or channelId" });
  }

  try {
    const template = WritingTemplatesService.getTemplate(templateType, context) as any;

    if (template.error) {
      return res.status(404).json(template);
    }

    const messageText = `📝 **${template.name} Template**\n\n${template.structure}\n\n---\n**Pro Tips:**\n${template.pro_tips?.map((t: string) => `• ${t}`).join("\n")}`;

    const channel = serverClient.channel(channelType, channelId);
    await channel.sendMessage({
      text: messageText,
      ai_generated: true,
    });

    res.json({ success: true, template });
  } catch (error) {
    res.status(500).json({
      error: "Failed to apply template",
      reason: (error as Error).message,
    });
  }
});

// ─── Chat Summary ──────────────────────────────────────────────────────────────
/**
 * POST /summarize-chat
 * Generate an AI summary of the chat conversation history
 */
app.post("/summarize-chat", async (req, res) => {
  const { channelId, channelType = "messaging", messageLimit = 50 } =
    req.body as ChatSummaryRequest;

  if (!channelId) {
    return res.status(400).json({ error: "Missing channelId" });
  }

  console.log(`[API] /summarize-chat → channel: ${channelId}`);

  try {
    // Fetch recent messages from Stream Chat
    const channel = serverClient.channel(channelType, channelId);
    await channel.watch();

    const response = await channel.query({
      messages: { limit: messageLimit },
    });

    const messages = response.messages || [];

    if (messages.length === 0) {
      return res.json({
        success: true,
        summary: "No messages found in this conversation.",
        messageCount: 0,
      });
    }

    // Format messages for AI
    const conversationText = messages
      .filter((m) => m.text && m.text.trim())
      .map((m) => {
        const role = m.user?.id?.startsWith("ai-bot") ? "AI Assistant" : m.user?.name || "User";
        return `${role}: ${m.text}`;
      })
      .join("\n\n");

    // Generate summary with Gemini
    const gemini = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });

    const summaryResponse = await gemini.models.generateContent({
      model: "gemini-2.0-flash",
      contents: [
        {
          role: "user",
          parts: [
            {
              text: `Please provide a comprehensive summary of this writing session conversation. Include:
1. **Main Topics Discussed**: What writing tasks were worked on
2. **Key Decisions Made**: Any important choices about style, content, or approach  
3. **Content Created**: What was written or edited
4. **Writing Progress**: How the work evolved through the session
5. **Next Steps**: Any mentioned follow-ups or unfinished tasks

Keep the summary organized, professional, and useful for continuing the work later.

---
CONVERSATION:
${conversationText}`,
            },
          ],
        },
      ],
      config: {
        temperature: 0.3,
        maxOutputTokens: 1024,
      },
    });

    const summary =
      summaryResponse.candidates?.[0]?.content?.parts?.[0]?.text ||
      "Unable to generate summary.";

    // Send summary as an AI message in the channel
    await channel.sendMessage({
      text: `📊 **Session Summary**\n\n${summary}`,
      ai_generated: true,
    });

    res.json({
      success: true,
      summary,
      messageCount: messages.length,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("[API] Chat summary failed:", error);
    res.status(500).json({
      error: "Failed to generate summary",
      reason: (error as Error).message,
    });
  }
});

// ─── Writing Analysis ──────────────────────────────────────────────────────────
/**
 * POST /analyze-writing
 * Analyze text for readability, tone, grammar issues
 */
app.post("/analyze-writing", async (req, res) => {
  const { text, channelId, channelType = "messaging" } = req.body;

  if (!text) {
    return res.status(400).json({ error: "Missing text" });
  }

  try {
    const gemini = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });

    const analysisResponse = await gemini.models.generateContent({
      model: "gemini-2.0-flash",
      contents: [
        {
          role: "user",
          parts: [
            {
              text: `Analyze the following text and provide a detailed writing analysis in this exact JSON format:
{
  "overall_score": <1-10>,
  "readability": {
    "score": <1-10>,
    "level": "<Very Easy|Easy|Standard|Fairly Difficult|Difficult|Very Difficult>",
    "feedback": "<specific feedback>"
  },
  "tone": {
    "primary": "<professional|casual|academic|persuasive|creative|technical>",
    "sentiment": "<positive|neutral|negative|mixed>",
    "feedback": "<specific feedback>"
  },
  "grammar": {
    "score": <1-10>,
    "issues_found": <number>,
    "issues": ["<issue 1>", "<issue 2>"]
  },
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"],
  "stats": {
    "word_count": <number>,
    "sentence_count": <number>,
    "avg_sentence_length": <number>,
    "reading_time_minutes": <number>
  }
}

Return ONLY the JSON, no other text.

TEXT TO ANALYZE:
${text}`,
            },
          ],
        },
      ],
      config: { temperature: 0.1, maxOutputTokens: 1024 },
    });

    const rawResponse =
      analysisResponse.candidates?.[0]?.content?.parts?.[0]?.text || "{}";

    let analysisData;
    try {
      const cleanJson = rawResponse.replace(/```json\n?|\n?```/g, "").trim();
      analysisData = JSON.parse(cleanJson);
    } catch {
      analysisData = { error: "Failed to parse analysis", raw: rawResponse };
    }

    // Format as readable message and send to channel
    if (channelId && analysisData && !analysisData.error) {
      const analysisMessage = `📊 **Writing Analysis**

**Overall Score:** ${analysisData.overall_score}/10

**Readability:** ${analysisData.readability?.level} (${analysisData.readability?.score}/10)
${analysisData.readability?.feedback}

**Tone:** ${analysisData.tone?.primary} • ${analysisData.tone?.sentiment}
${analysisData.tone?.feedback}

**Grammar Score:** ${analysisData.grammar?.score}/10 (${analysisData.grammar?.issues_found} issues found)
${analysisData.grammar?.issues?.map((i: string) => `• ${i}`).join("\n") || "No major issues"}

**Strengths:**
${analysisData.strengths?.map((s: string) => `✅ ${s}`).join("\n")}

**Areas to Improve:**
${analysisData.improvements?.map((i: string) => `💡 ${i}`).join("\n")}

**Stats:** ${analysisData.stats?.word_count} words • ${analysisData.stats?.sentence_count} sentences • ~${analysisData.stats?.reading_time_minutes} min read`;

      const channel = serverClient.channel(channelType, channelId);
      await channel.sendMessage({
        text: analysisMessage,
        ai_generated: true,
      });
    }

    res.json({ success: true, analysis: analysisData });
  } catch (error) {
    res.status(500).json({
      error: "Writing analysis failed",
      reason: (error as Error).message,
    });
  }
});

// ─── Helpers ───────────────────────────────────────────────────────────────────
async function disposeAiAgent(aiAgent: AIAgent) {
  await aiAgent.dispose();
  if (!aiAgent.user) return;
  await serverClient.deleteUser(aiAgent.user.id, { hard_delete: true });
}

// ─── Start Server ──────────────────────────────────────────────────────────────
const port = process.env.PORT || 3333;
app.listen(port, () => {
  console.log(`\n🚀 AI Writing Assistant Server v2.0`);
  console.log(`📍 Running on http://localhost:${port}`);
  console.log(`🤖 Provider: Google Gemini 2.0 Flash (Free)`);
  console.log(`🔍 Search: Google Grounding + DuckDuckGo`);
  console.log(`📄 Features: Document Analysis, Templates, Chat Summary\n`);
});