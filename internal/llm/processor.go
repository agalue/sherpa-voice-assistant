package llm

import (
	"context"
	"log"
)

// RunProcessor reads user transcriptions from in, generates LLM responses via Chat,
// and sends them to out. It is intended to be run as a goroutine and returns when
// ctx is cancelled or in is closed.
func (c *Client) RunProcessor(ctx context.Context, in <-chan string, out chan<- string) {
	for {
		select {
		case <-ctx.Done():
			return
		case text, ok := <-in:
			if !ok {
				return
			}

			log.Printf("🧠 Processing: %q", text)

			response, err := c.Chat(ctx, text)
			if err != nil {
				log.Printf("❌ LLM error: %v", err)
				select {
				case out <- "I'm sorry, I encountered an error.":
				case <-ctx.Done():
					return
				}
				continue
			}

			log.Printf("🤖 Assistant: %s", response)

			select {
			case out <- response:
			case <-ctx.Done():
				return
			}
		}
	}
}
