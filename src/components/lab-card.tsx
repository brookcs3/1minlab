import type { ReactNode } from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

type LabCardProps = {
  labNumber: number;
  title: string;
  description: string;
  icon: ReactNode;
  children: ReactNode;
};

export function LabCard({
  labNumber,
  title,
  description,
  icon,
  children,
}: LabCardProps) {
  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
            {icon}
          </div>
          <div className="flex-1">
            <CardTitle className="font-headline text-xl">
              Lab {labNumber}: {title}
            </CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="item-1">
            <AccordionTrigger className="text-accent-foreground hover:no-underline">View Write-Up</AccordionTrigger>
            <AccordionContent>
              <div className="prose max-w-none pt-4">
                {children}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}
