import { useEffect, useState } from 'react';
import { IconDown, IconRight } from '@arco-design/web-react/icon';

type Props = {
  text: string;
  streaming?: boolean;
  defaultOpen?: boolean;
};

export default function ThinkingBlock({ text, streaming, defaultOpen }: Props) {
  const [open, setOpen] = useState(Boolean(defaultOpen || streaming));

  useEffect(() => {
    if (streaming) setOpen(true);
    else if (text) {
      // auto-collapse shortly after thinking finishes
      const t = window.setTimeout(() => setOpen(false), 400);
      return () => window.clearTimeout(t);
    }
  }, [streaming, text]);

  if (!text && !streaming) return null;

  return (
    <div className={`think-block ${open ? 'open' : ''} ${streaming ? 'streaming' : ''}`}>
      <button type="button" className="think-toggle" onClick={() => setOpen((v) => !v)}>
        <span className="think-chevron">{open ? <IconDown /> : <IconRight />}</span>
        <span className="think-label">
          {streaming ? '正在深度思考…' : '深度思考'}
        </span>
        {!streaming && text && (
          <span className="think-meta">{Math.max(1, Math.round(text.length / 40))} 秒可读</span>
        )}
      </button>
      {open && (
        <div className="think-body">
          {text || (streaming ? '…' : '')}
        </div>
      )}
    </div>
  );
}
