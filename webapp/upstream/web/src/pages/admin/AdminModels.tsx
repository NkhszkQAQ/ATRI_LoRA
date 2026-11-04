import { useEffect, useState } from 'react';
import { Button, Input, Message, Space, Tag } from '@arco-design/web-react';
import { api } from '../../api';

export default function AdminModels() {
  const [active, setActive] = useState('');
  const [available, setAvailable] = useState<string[]>([]);
  const [custom, setCustom] = useState('');

  const load = async () => {
    const d = await api.adminModels();
    setActive(d.active);
    setAvailable(d.available || []);
  };

  useEffect(() => { load().catch((e) => Message.error(e.message)); }, []);

  const apply = async (model: string) => {
    await api.adminSetModel(model);
    Message.success(`已切换到底层模型 ${model}`);
    load();
  };

  return (
    <div>
      <h2>底层模型</h2>
      <p>当前激活：<Tag color="arcoblue">{active || '-'}</Tag></p>
      <p style={{ color: 'var(--muted)' }}>
        默认按显存自动选择（≤10GB → qwen3:4b）。此处可手动覆盖；需模型已在 Ollama 中拉取。
      </p>
      <Space wrap style={{ marginBottom: 16 }}>
        {available.map((m) => (
          <Button key={m} type={m === active ? 'primary' : 'outline'} onClick={() => apply(m)}>
            {m}
          </Button>
        ))}
      </Space>
      <Space>
        <Input placeholder="自定义标签，如 qwen3:4b" value={custom} onChange={setCustom} style={{ width: 240 }} />
        <Button type="primary" onClick={() => apply(custom)} disabled={!custom.trim()}>应用</Button>
        <Button onClick={load}>刷新</Button>
      </Space>
    </div>
  );
}
