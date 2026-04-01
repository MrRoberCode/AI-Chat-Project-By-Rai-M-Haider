import { Content, GoogleGenAI, Tool } from "@google/genai";
import type { Channel, Event, MessageResponse, StreamChat } from "stream-chat";
import { WritingTemplatesService } from "../services/WritingTemplatesService";
import { DuckDuckGoService } from "../services/DuckDuckGoService";

export class GeminiResponseHandler {
  private message_text = "";
  private is_done = false;
  private last_update_time = 0;

  constructor(
    private readonly gemini: GoogleGenAI,
    private readonly conversationHistory: Content[],
    private readonly systemPrompt: string,
    private readonly tools: Tool[],
    private readonly chatClient: StreamChat,
    private readonly channel: Channel,
    private readonly message: MessageResponse,
    private readonly onDispose: (assistantMessage: string) => void
  ) {
    this.chatClient.on("ai_indicator.stop", this.handleStopGenerating);
  }

  run = async () => {
    const { cid, id: message_id } = this.message;

    try {
      await this.channel.sendEvent({
        type: "ai_indicator.update",
        ai_state: "AI_STATE_GENERATING",
        cid,
        message_id,
      });

      const model = this.gemini.models;
      let continueLoop = true;
      let currentHistory = [...this.conversationHistory];

      while (continueLoop) {
        continueLoop = false;

        const stream = model.generateContentStream({
          model: "gemini-1.5-flash",
          contents: currentHistory,
          config: {
            systemInstruction: this.systemPrompt,
            temperature: 0.7,
            maxOutputTokens: 4096,
            tools: this.tools,
          },
        });

        let fullResponseText = "";
        let functionCalls: any[] = [];

        for await (const chunk of await stream) {
          if (this.is_done) break;

          const candidate = chunk.candidates?.[0];
          if (!candidate) continue;

          // Handle text content
          const textPart = candidate.content?.parts?.find((p: any) => p.text);
          if (textPart?.text) {
            fullResponseText += textPart.text;
            this.message_text = fullResponseText;

            // Throttle updates to 300ms
            const now = Date.now();
            if (now - this.last_update_time > 300) {
              await this.chatClient.partialUpdateMessage(message_id, {
                set: { text: this.message_text },
              });
              this.last_update_time = now;
            }
          }

          // Collect function calls
          const fnParts = candidate.content?.parts?.filter(
            (p: any) => p.functionCall
          );
          if (fnParts?.length) {
            functionCalls.push(...fnParts.map((p: any) => p.functionCall));
          }
        }

        // Handle function calls if any
        if (functionCalls.length > 0 && !this.is_done) {
          await this.channel.sendEvent({
            type: "ai_indicator.update",
            ai_state: "AI_STATE_EXTERNAL_SOURCES",
            cid,
            message_id,
          });

          const functionResults = await this.executeFunctionCalls(functionCalls);

          // Add model response and function results to history
          currentHistory.push({
            role: "model",
            parts: functionCalls.map((fc) => ({ functionCall: fc })),
          });

          currentHistory.push({
            role: "user",
            parts: functionResults.map((result) => ({
              functionResponse: result,
            })),
          });

          // Continue loop to get final response
          continueLoop = true;
          this.message_text = "";

          await this.channel.sendEvent({
            type: "ai_indicator.update",
            ai_state: "AI_STATE_GENERATING",
            cid,
            message_id,
          });
        }
      }

      if (!this.is_done) {
        // Final message update
        await this.chatClient.partialUpdateMessage(message_id, {
          set: { text: this.message_text },
        });

        await this.channel.sendEvent({
          type: "ai_indicator.clear",
          cid,
          message_id,
        });
      }
    } catch (error) {
      console.error("[GeminiResponseHandler] Error:", error);
      await this.handleError(error as Error);
    } finally {
      await this.dispose();
    }
  };

  private executeFunctionCalls = async (functionCalls: any[]) => {
    const results = [];

    for (const fc of functionCalls) {
      console.log(`[GeminiResponseHandler] Executing function: ${fc.name}`);

      try {
        let output: any;

        switch (fc.name) {
          case "duckduckgo_search":
            output = await DuckDuckGoService.search(fc.args.query);
            break;

          case "analyze_writing":
            output = this.analyzeWriting(fc.args.text);
            break;

          case "get_writing_template":
            output = WritingTemplatesService.getTemplate(
              fc.args.template_type,
              fc.args.context
            );
            break;

          case "summarize_conversation":
            output = this.summarizeConversation(fc.args.focus);
            break;

          default:
            output = { error: `Unknown function: ${fc.name}` };
        }

        results.push({
          name: fc.name,
          response: { output: JSON.stringify(output) },
        });
      } catch (error) {
        console.error(`[GeminiResponseHandler] Function ${fc.name} failed:`, error);
        results.push({
          name: fc.name,
          response: { output: JSON.stringify({ error: "Function execution failed" }) },
        });
      }
    }

    return results;
  };

  private analyzeWriting = (text: string) => {
    const words = text.trim().split(/\s+/).length;
    const sentences = text.split(/[.!?]+/).filter((s) => s.trim()).length;
    const avgWordsPerSentence = sentences > 0 ? Math.round(words / sentences) : 0;
    const paragraphs = text.split(/\n\n+/).filter((p) => p.trim()).length;

    // Simple readability score (Flesch-Kincaid approximation)
    const syllables = text
      .toLowerCase()
      .replace(/[^a-z]/g, " ")
      .split(/\s+/)
      .reduce((count, word) => {
        const syllableCount = word
          .replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, "")
          .replace(/^y/, "")
          .match(/[aeiouy]{1,2}/g);
        return count + (syllableCount ? syllableCount.length : 1);
      }, 0);

    const fleschScore =
      sentences > 0 && words > 0
        ? Math.round(
            206.835 -
              1.015 * (words / sentences) -
              84.6 * (syllables / words)
          )
        : 0;

    const readabilityLevel =
      fleschScore >= 80
        ? "Very Easy"
        : fleschScore >= 70
        ? "Easy"
        : fleschScore >= 60
        ? "Standard"
        : fleschScore >= 50
        ? "Fairly Difficult"
        : fleschScore >= 30
        ? "Difficult"
        : "Very Difficult";

    return {
      stats: {
        word_count: words,
        sentence_count: sentences,
        paragraph_count: paragraphs,
        avg_words_per_sentence: avgWordsPerSentence,
        reading_time_minutes: Math.ceil(words / 200),
      },
      readability: {
        score: Math.max(0, Math.min(100, fleschScore)),
        level: readabilityLevel,
      },
      suggestions:
        avgWordsPerSentence > 25
          ? ["Consider breaking long sentences into shorter ones for better readability"]
          : avgWordsPerSentence < 8
          ? ["Your sentences are very short — consider combining some for better flow"]
          : ["Sentence length looks good"],
    };
  };

  private summarizeConversation = (focus: string) => {
    const messageCount = this.conversationHistory.length;
    const userMessages = this.conversationHistory
      .filter((m) => m.role === "user")
      .map((m) => m.parts?.[0]?.text || "")
      .filter(Boolean);

    return {
      total_exchanges: Math.floor(messageCount / 2),
      focus,
      user_topics: userMessages.slice(-5),
      session_start: new Date().toISOString(),
    };
  };

  dispose = async () => {
    if (this.is_done) return;
    this.is_done = true;
    this.chatClient.off("ai_indicator.stop", this.handleStopGenerating);
    this.onDispose(this.message_text);
  };

  private handleStopGenerating = async (event: Event) => {
    if (this.is_done || event.message_id !== this.message.id) return;

    console.log("[GeminiResponseHandler] Stop generating:", this.message.id);
    this.is_done = true;

    await this.chatClient.partialUpdateMessage(this.message.id, {
      set: { text: this.message_text || "Generation stopped." },
    });

    await this.channel.sendEvent({
      type: "ai_indicator.clear",
      cid: this.message.cid,
      message_id: this.message.id,
    });

    await this.dispose();
  };

  private handleError = async (error: Error) => {
    if (this.is_done) return;

    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_ERROR",
      cid: this.message.cid,
      message_id: this.message.id,
    });

    await this.chatClient.partialUpdateMessage(this.message.id, {
      set: {
        text: `Sorry, I encountered an error: ${error.message ?? "Unknown error"}. Please try again.`,
      },
    });
  };
}