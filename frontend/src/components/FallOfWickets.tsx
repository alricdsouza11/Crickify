import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface FallOfWicketsProps {
  wickets: string[];
}

export default function FallOfWickets({ wickets }: FallOfWicketsProps) {
  if (!wickets || wickets.length === 0) return null;

  return (
    <motion.div
      className="mt-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
    >
      <Card className="border-none shadow-sm bg-white dark:bg-gray-900 overflow-hidden">
        <CardHeader className="bg-blue-50 dark:bg-blue-900/20 border-b border-blue-100 dark:border-blue-800 py-3">
          <CardTitle className="text-blue-700 dark:text-blue-300 text-base font-medium">
            Fall of wickets
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <motion.div
            className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.3 }}
          >
            {wickets.map((wicket, index) => (
              <motion.span
                key={index}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.1 * index }}
                className="inline-block"
              >
                <span>{wicket}</span>
                {index < wickets.length - 1 && (
                  <span className="inline-flex items-center px-2 text-blue-400">
                    <span className="h-1 w-1 rounded-full bg-blue-400 mx-1"></span>
                  </span>
                )}
              </motion.span>
            ))}
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
