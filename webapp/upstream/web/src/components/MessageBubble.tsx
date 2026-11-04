import ThinkingBlock from './ThinkingBlock';

type Props = {
  role: string;
  content: string;
  reasoning?: string;
  streaming?: boolean;
  thinkingStreaming?: boolean;
};

export default function MessageBubble({
  role,
  content,
  reasoning,
  streaming,
  thinkingStreaming,
}: Props) {
  if (role === 'user') {
    return (
      <div className="msg-row user">
        <div className="msg-user">{content}</div>
      </div>
    );
  }

  return (
    <div className="msg-row assistant">
      <div className="msg-assistant">
        <div className="msg-role">Atri</div>
        <ThinkingBlock
          text={reasoning || ''}
          streaming={thinkingStreaming}
          defaultOpen={false}
        />
        <div className="msg-content">
          {content || (streaming && !thinkingStreaming ? '…' : '')}
        </div>
      </div>
    </div>
  );
}
