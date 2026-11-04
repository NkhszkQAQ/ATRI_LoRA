import { useEffect, useState } from 'react';
import { Message, Table, Typography } from '@arco-design/web-react';
import dayjs from 'dayjs';
import { api } from '../../api';

export default function AdminAudit() {
  const [rows, setRows] = useState<any[]>([]);
  useEffect(() => {
    api.adminAudit().then(setRows).catch((e) => Message.error(e.message));
  }, []);

  return (
    <div>
      <h2>审计日志</h2>
      <Table
        rowKey="id"
        data={rows}
        scroll={{ x: 1000 }}
        columns={[
          {
            title: '操作者',
            dataIndex: 'actor_name',
            width: 150,
            render: (v, r) => (
              <div>
                <div>{v || '系统'}</div>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {r.actor_email}
                </Typography.Text>
              </div>
            ),
          },
          { title: '动作', dataIndex: 'action', width: 180 },
          {
            title: '目标',
            render: (_, r) => `${r.target_type || '—'}${r.target_id ? `:${r.target_id}` : ''}`,
          },
          { title: 'IP', dataIndex: 'ip', width: 120, ellipsis: true },
          {
            title: 'UA',
            dataIndex: 'user_agent',
            width: 180,
            ellipsis: true,
            render: (v) => v || '—',
          },
          {
            title: '时间',
            dataIndex: 'created_at',
            width: 170,
            render: (v) => dayjs(v).format('YYYY-MM-DD HH:mm:ss'),
          },
        ]}
      />
    </div>
  );
}
