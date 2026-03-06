import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  webpack(config, { dev, isServer }) {
    if (dev && !isServer) {
      config.watchOptions = {
        poll: 1000,         // ตรวจสอบการเปลี่ยนแปลงไฟล์ทุก 1 วินาที (แก้ปัญหา Docker volume ไม่แจ้งเตือน)
        aggregateTimeout: 300, // หน่วงเวลาเล็กน้อยหลังจากเปลี่ยนไฟล์ก่อนจะ rebuild
      };

      // ตั้งค่า HMR Client ให้เชื่อมต่อผ่านโดเมนจริง (teamder.online) 
      // แทนที่จะพยายามต่อไปที่ localhost:8080 ของเบราว์เซอร์
      config.entry = config.entry().then((entry: any) => {
        return Object.keys(entry).reduce((acc, key) => {
          acc[key] = entry[key].map((path: string) => {
            if (path.includes('webpack-hot-middleware/client')) {
              // บอกให้มันวิ่งไปหาหน้าบ้านผ่าน HTTPS/WSS
              return `${path}&path=/_next/webpack-hmr&timeout=20000&reload=true`;
            }
            return path;
          });
          return acc;
        }, {} as any);
      });
    }
    return config;
  },
  
  // เพิ่มการอนุญาตให้เชื่อมต่อ WebSocket จากโดเมนภายนอก
  async headers() {
    return [
      {
        source: "/_next/webpack-hmr",
        headers: [
          { key: "Upgrade", value: "websocket" },
        ],
      },
    ];
  },
};

export default nextConfig;