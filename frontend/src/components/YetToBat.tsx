import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface YetToBatProps {
  players: string[];
}

export default function YetToBat({ players }: YetToBatProps) {
  if (!players || players.length === 0) return null;

  return (
    <motion.div
      className="mt-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <Card className="border-none shadow-sm bg-white dark:bg-gray-900 overflow-hidden">
        <CardHeader className="bg-blue-50 dark:bg-blue-900/20 border-b border-blue-100 dark:border-blue-800 py-3">
          <CardTitle className="text-blue-700 dark:text-blue-300 text-base font-medium">
            Yet to bat
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <motion.div
            className="flex flex-wrap gap-2"
            variants={{
              hidden: { opacity: 0 },
              visible: {
                opacity: 1,
                transition: { staggerChildren: 0.05 },
              },
            }}
            initial="hidden"
            animate="visible"
          >
            {players.map((player, index) => (
              <motion.span
                key={index}
                variants={{
                  hidden: { opacity: 0, y: 10 },
                  visible: { opacity: 1, y: 0 },
                }}
                className="inline-flex items-center px-3 py-1 rounded-full bg-blue-50 dark:bg-blue-900/20 text-sm text-blue-700 dark:text-blue-300 border border-blue-100 dark:border-blue-800"
                whileHover={{
                  scale: 1.05,
                  backgroundColor: "rgba(59, 130, 246, 0.2)",
                  transition: { duration: 0.2 },
                }}
              >
                {player}
              </motion.span>
            ))}
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
