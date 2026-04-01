/**
 * DuckDuckGo Search Service
 * Completely free, no API key required
 * Uses DuckDuckGo's instant answer API + HTML scraping approach
 */
export class DuckDuckGoService {
  private static readonly DDG_API = "https://api.duckduckgo.com/";

  static search = async (query: string): Promise<object> => {
    console.log(`[DuckDuckGo] Searching for: "${query}"`);

    try {
      // DuckDuckGo Instant Answer API (free, no key needed)
      const params = new URLSearchParams({
        q: query,
        format: "json",
        no_redirect: "1",
        no_html: "1",
        skip_disambig: "1",
      });

      const response = await fetch(`${this.DDG_API}?${params}`, {
        headers: {
          "User-Agent": "AI-Writing-Assistant/1.0",
        },
      });

      if (!response.ok) {
        throw new Error(`DuckDuckGo API error: ${response.status}`);
      }

      const data = await response.json();

      // Build structured results
      const results: any[] = [];

      // Abstract (main answer)
      if (data.Abstract) {
        results.push({
          title: data.Heading || query,
          snippet: data.Abstract,
          url: data.AbstractURL || "",
          source: data.AbstractSource || "DuckDuckGo",
          type: "abstract",
        });
      }

      // Answer (direct answer like "the capital of France is Paris")
      if (data.Answer) {
        results.push({
          title: "Direct Answer",
          snippet: data.Answer,
          url: "",
          source: "DuckDuckGo",
          type: "answer",
        });
      }

      // Related topics
      if (data.RelatedTopics?.length) {
        const topics = data.RelatedTopics.slice(0, 5)
          .filter((t: any) => t.Text && !t.Topics) // Skip category groups
          .map((t: any) => ({
            title: t.FirstURL?.split("/").pop()?.replace(/_/g, " ") || "",
            snippet: t.Text,
            url: t.FirstURL || "",
            source: "DuckDuckGo",
            type: "related",
          }));
        results.push(...topics);
      }

      // Infobox data
      if (data.Infobox?.content?.length) {
        const infoItems = data.Infobox.content
          .slice(0, 5)
          .map((item: any) => `${item.label}: ${item.value}`)
          .join("; ");

        if (infoItems) {
          results.push({
            title: `${data.Heading} - Key Facts`,
            snippet: infoItems,
            url: "",
            source: "DuckDuckGo Infobox",
            type: "facts",
          });
        }
      }

      if (results.length === 0) {
        return {
          query,
          message: "No instant answers found. The topic may require more specific search terms.",
          suggestion: `Try searching on Google: https://www.google.com/search?q=${encodeURIComponent(query)}`,
          results: [],
        };
      }

      return {
        query,
        result_count: results.length,
        results,
        search_time: new Date().toISOString(),
      };
    } catch (error) {
      console.error("[DuckDuckGo] Search error:", error);
      return {
        query,
        error: "Search failed",
        message: error instanceof Error ? error.message : "Unknown error",
        results: [],
      };
    }
  };
}