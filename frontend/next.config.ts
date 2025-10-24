import { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "api",
        pathname: "**",
      },
    ],
  },
  swcMinify: true,
};

export default config;
