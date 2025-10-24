import { useTheme } from "next-themes";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { SunIcon, MoonIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function ThemeToggle() {
  const [mounted, setMounted] = useState(false);
  const { theme, setTheme } = useTheme();

  // After mounting, we have access to the theme
  useEffect(() => setMounted(true), []);

  if (!mounted) return null;

  const isDark = theme === "dark";

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Button
        variant="outline"
        size="icon"
        onClick={() => setTheme(isDark ? "light" : "dark")}
        className="cursor-pointer relative w-10 h-10 rounded-full bg-white dark:bg-gray-800 border-blue-100 dark:border-blue-800 hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-colors duration-300"
        aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      >
        <motion.div
          initial={false}
          animate={{
            rotate: isDark ? 0 : 180,
            scale: isDark ? 1 : 0,
          }}
          transition={{ duration: 0.5, type: "spring", stiffness: 200 }}
          style={{ position: "absolute", transformOrigin: "center" }}
        >
          <MoonIcon
            className="h-5 w-5 text-blue-600 dark:text-blue-400"
            strokeWidth={1.5}
          />
        </motion.div>

        <motion.div
          initial={false}
          animate={{
            rotate: isDark ? 180 : 0,
            scale: isDark ? 0 : 1,
          }}
          transition={{ duration: 0.5, type: "spring", stiffness: 200 }}
          style={{ position: "absolute", transformOrigin: "center" }}
        >
          <SunIcon
            className="h-5 w-5 text-blue-600 dark:text-blue-400"
            strokeWidth={1.5}
          />
        </motion.div>
      </Button>
    </motion.div>
  );
}
