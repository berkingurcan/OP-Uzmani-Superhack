import clsx from 'clsx'
import Balancer from 'react-wrap-balancer'
import ReactMarkdown from 'react-markdown'
import { CopyBlock, CodeBlock } from "react-code-blocks";

// wrap Balancer to remove type errors :( - @TODO - fix this ugly hack
const BalancerWrapper = (props: any) => <Balancer {...props} />

type ChatGPTAgent = 'user' | 'system' | 'assistant'

export interface ChatGPTMessage {
  role: ChatGPTAgent
  content: string
}

// loading placeholder animation for the chat line
export const LoadingChatLine = () => (
  <div className="flex min-w-full animate-pulse px-4 py-5 sm:px-6">
    <div className="flex flex-grow space-x-3">
      <div className="min-w-0 flex-1">
        <p className="font-large text-xxl text-gray-900">
          <a href="#" className="hover:underline">
            AI
          </a>
        </p>
        <div className="space-y-4 pt-4">
          <div className="grid grid-cols-3 gap-4">
            <div className="col-span-2 h-2 rounded bg-zinc-500"></div>
            <div className="col-span-1 h-2 rounded bg-zinc-500"></div>
          </div>
          <div className="h-2 rounded bg-zinc-500"></div>
        </div>
      </div>
    </div>
  </div>
)

// util helper to convert new lines to <br /> tags
const convertNewLines = (text: string) => {
  const lines = text.split('\n');
  const result = [];
  let codeBlock = false;
  let codeLines = [];

  for (let line of lines) {
    if (line.trim().startsWith('```')) {
      if (codeBlock) {
        // Close the code block
        result.push(
          <div key={result.length} className="codebox">
            <CodeBlock
              text={codeLines.join('\n')}
              language={'ts'}
              showLineNumbers={false}
              wrapLines
            />
          </div>
        );
        codeLines = [];
        codeBlock = false;
      } else {
        // Start a new code block
        codeBlock = true;
      }
    } else if (codeBlock) {
      // Inside a code block, just collect the lines
      codeLines.push(line);
    } else {
      // Regular text lines
      result.push(
        <span key={result.length}>
          {line}
          <br />
        </span>
      );
    }
  }

  // Check if there's an unclosed code block
  if (codeBlock && codeLines.length > 0) {
    result.push(
      <div key={result.length} className="codebox">
        <CodeBlock
              text={codeLines.join('\n')}
              language={'ts'}
              showLineNumbers={false}
              wrapLines
            />
      </div>
    );
  }

  return result;
};


export function ChatLine({ role = 'assistant', content }: ChatGPTMessage) {
  if (!content) {
    return null
  }
  const formatteMessage = convertNewLines(content)

  return (
    <div
      className={
        role != 'assistant' ? 'float-right clear-both' : 'float-left clear-both'
      }
    >
      <BalancerWrapper>
        <div className="float-right mb-5 rounded-lg bg-white px-4 py-5 shadow-lg ring-1 ring-zinc-100 sm:px-6">
          <div className="flex space-x-3">
            <div className="flex-1 gap-4">
              <p className="font-large text-xxl text-gray-900">
                <a href="#" className="hover:underline">
                  {role == 'assistant' ? 'AI' : 'You'}
                </a>
              </p>
              <p
                className={clsx(
                  'text ',
                  role == 'assistant' ? 'font-semibold font- ' : 'text-gray-400'
                )}
              >
              {formatteMessage}
              </p>
            </div>
          </div>
        </div>
      </BalancerWrapper>
    </div>
  )
}
