import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button, Input, Message, Switch, Tooltip } from '@arco-design/web-react';
import { IconPlus, IconSend } from '@arco-design/web-react/icon';
import { api, clearToken, streamChat } from '../api';
import MessageBubble from '../components/MessageBubble';

type Msg = {
  role: string;
  content: string;
  reasoning?: string;
};

export default function ChatPage() {
  const [me, setMe] = useState<any>(null);
  const [convs, setConvs] = useState<any[]>([]);
  const [cid, setCid] = useState<number | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [thinking, setThinking] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshConvs = async () => {
    const list = await api.conversations();
    setConvs(list);
  };

  useEffect(() => {
    (async () => {
      try {
        setMe(await api.me());
        await refreshConvs();
      } catch {
        clearToken();
        location.href = '/login';
      }
    })();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [msgs, busy]);

  const openConv = async (id: number) => {
    setCid(id);
    const list = await api.messages(id);
    setMsgs(
      list.map((m: any) => ({
        role: m.role,
        content: m.content,
        reasoning: m.meta?.reasoning || '',
      })),
    );
  };

  const newChat = async () => {
    const c = await api.createConversation();
    await refreshConvs();
    setCid(c.id);
    setMsgs([]);
  };

  const send = async () => {
    const content = text.trim();
    if (!content || busy) return;
    setText('');
    setMsgs((m) => [...m, { role: 'user', content }, { role: 'assistant', content: '', reasoning: '' }]);
    setBusy(true);
    try {
      await streamChat(content, cid, thinking, (data) => {
        if (data.conversation_id && data.conversation_id !== cid) setCid(data.conversation_id);
        if (data.content || data.reasoning) {
          setMsgs((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === 'assistant') {
              next[next.length - 1] = {
                ...last,
                content: last.content + (data.content || ''),
                reasoning: (last.reasoning || '') + (data.reasoning || ''),
              };
            }
            return next;
          });
        }
      });
      await refreshConvs();
    } catch (e: any) {
      Message.error(e.message || '发送失败');
    } finally {
      setBusy(false);
    }
  };

  const last = msgs[msgs.length - 1];
  const thinkingStreaming =
    busy && last?.role === 'assistant' && !last.content && (thinking || !!last.reasoning);
  const contentStreaming = busy && last?.role === 'assistant';

  return (
    <div className={`shell ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
      <aside className="sidebar">
        <div className="sidebar-top">
          <div className="brand">Atri</div>
          <Button type="outline" icon={<IconPlus />} onClick={newChat} long className="new-chat-btn">
            新对话
          </Button>
        </div>
        <div className="conv-list">
          {convs.map((c) => (
            <div
              key={c.id}
              className={`conv-item ${cid === c.id ? 'active' : ''}`}
              onClick={() => openConv(c.id)}
            >
              <div className="conv-title">{c.title || '未命名'}</div>
              <div className="conv-meta">{c.model || 'atri-lora'}</div>
            </div>
          ))}
        </div>
        <div className="sidebar-foot">
          <div className="user-line">{me?.display_name || me?.email}</div>
          {me?.role === 'admin' && (
            <Link to="/admin" className="side-link">管理后台</Link>
          )}
          <a
            href="#"
            className="side-link"
            onClick={(e) => {
              e.preventDefault();
              clearToken();
              location.href = '/login';
            }}
          >
            退出
          </a>
        </div>
      </aside>

      <main className="main">
        <header className="chat-topbar">
          <button type="button" className="ghost-btn" onClick={() => setSidebarOpen((v) => !v)}>
            {sidebarOpen ? '收起侧栏' : '展开侧栏'}
          </button>
          <div className="topbar-title">ATRI LoRA</div>
          <div className="topbar-right">
            <Tooltip content="开启后模型会先推理再回答，可折叠查看">
              <span className="think-switch">
                深度思考
                <Switch checked={thinking} onChange={setThinking} size="small" />
              </span>
            </Tooltip>
          </div>
        </header>

        <div className="messages">
          {!msgs.length && (
            <div className="hero-empty">
              <div className="brand hero-brand">Atri</div>
              <p className="hero-sub">高性能陪伴对话。非官方同人，非商业用途。</p>
              <div className="hero-hints">
                {['早安呀', '今天有点累', '教我做简单的饭'].map((h) => (
                  <button key={h} type="button" className="hint-chip" onClick={() => setText(h)}>
                    {h}
                  </button>
                ))}
              </div>
            </div>
          )}
          {msgs.map((m, i) => {
            const isLast = i === msgs.length - 1;
            return (
              <MessageBubble
                key={i}
                role={m.role}
                content={m.content}
                reasoning={m.reasoning}
                streaming={isLast && contentStreaming}
                thinkingStreaming={isLast && thinkingStreaming}
              />
            );
          })}
          <div ref={bottomRef} />
        </div>

        <div className="composer">
          <div className="composer-inner">
            <Input.TextArea
              value={text}
              onChange={setText}
              autoSize={{ minRows: 2, maxRows: 8 }}
              placeholder="给 Atri 发消息…"
              onPressEnter={(e) => {
                if (!e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <div className="composer-bar">
              <span className="composer-hint">Enter 发送 · Shift+Enter 换行</span>
              <Button
                type="primary"
                icon={<IconSend />}
                loading={busy}
                onClick={send}
                className="send-btn"
              >
                发送
              </Button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
