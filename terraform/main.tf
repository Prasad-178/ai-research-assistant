# This block is now in providers.tf
# provider "aws" {
#   region = var.aws_region
# }

# --- Unique Suffix for Naming ---
# This ensures that resource names are unique for each deployment,
# preventing collisions from failed pipeline runs. We use a short random ID
# to stay within the 32-character name limit for some AWS resources.
resource "random_id" "suffix" {
  byte_length = 4 # Creates an 8-character hex string
}

# --- Networking (VPC and Subnets) ---
# Using default VPC and subnets for simplicity.
# For production, you would create a dedicated VPC.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# --- ECS Cluster ---
resource "aws_ecs_cluster" "ml_app_cluster" {
  name = "${var.project_name}-cluster"
}

# --- Application Load Balancer (ALB) ---
# This will expose our service to the internet on port 80.
resource "aws_lb" "ml_app_lb" {
  name               = "${var.project_name_short}-lb-${random_id.suffix.hex}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb_sg.id]
  subnets            = data.aws_subnets.default.ids

  tags = {
    Name = "${var.project_name}-lb"
  }
}

resource "aws_lb_target_group" "ml_app_tg" {
  name        = "${var.project_name_short}-tg-${random_id.suffix.hex}"
  port        = 8000 # The port the container listens on
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    path                = "/" # Use the simple root endpoint for health checks
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.ml_app_lb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ml_app_tg.arn
  }
}

# --- Security Groups ---
# Security group for the Load Balancer
resource "aws_security_group" "lb_sg" {
  name        = "${var.project_name_short}-lb-sg-${random_id.suffix.hex}"
  description = "Allow HTTP traffic to Load Balancer"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    protocol    = "tcp"
    from_port   = 80
    to_port     = 80
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security group for the ECS Tasks
resource "aws_security_group" "ecs_tasks_sg" {
  name        = "${var.project_name_short}-tasks-sg-${random_id.suffix.hex}"
  description = "Allow traffic from the Load Balancer to the ECS tasks"
  vpc_id      = data.aws_vpc.default.id

  # Allow inbound traffic from the ALB on the container port
  ingress {
    protocol        = "tcp"
    from_port       = 8000
    to_port         = 8000
    security_groups = [aws_security_group.lb_sg.id]
  }

  # Allow all outbound traffic
  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- IAM Roles ---
# Role for ECS tasks to access other AWS services (like S3)
resource "aws_iam_role" "ecs_task_role" {
  name = "${var.project_name_short}-ecs-task-role-${random_id.suffix.hex}"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })
}

# Policy allowing tasks to access the S3 model bucket
resource "aws_iam_policy" "s3_access_policy" {
  name = "${var.project_name_short}-s3-access-policy-${random_id.suffix.hex}"
  policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = [
          "s3:GetObject",
          "s3:ListBucket"
        ],
        Resource = [
          "arn:aws:s3:::${var.model_s3_bucket}",
          "arn:aws:s3:::${var.model_s3_bucket}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "task_s3_access" {
  role       = aws_iam_role.ecs_task_role.name
  policy_arn = aws_iam_policy.s3_access_policy.arn
}

# Role for ECS to execute tasks (pulling images, etc.)
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "${var.project_name_short}-ecs-task-execution-role-${random_id.suffix.hex}"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- CloudWatch Log Group ---
resource "aws_cloudwatch_log_group" "ml_app_logs" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 7
}

# --- ECS Task Definition & Service ---
resource "aws_ecs_task_definition" "ml_app_task" {
  family                   = "${var.project_name}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  task_role_arn            = aws_iam_role.ecs_task_role.arn
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn

  ephemeral_storage {
    size_in_gib = 50 # Increase ephemeral storage to 50GB
  }

  container_definitions = jsonencode([
    {
      name      = "${var.project_name}-container"
      image     = var.ecr_image_uri
      cpu       = var.ecs_task_cpu
      memory    = var.ecs_task_memory
      essential = true
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
        }
      ]
      environment = [
        {
          name  = "MODEL_S3_BUCKET"
          value = var.model_s3_bucket
        },
        {
          name  = "MODEL_S3_KEY"
          value = var.model_s3_key_prefix
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ml_app_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "ml_app_service" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.ml_app_cluster.id
  task_definition = aws_ecs_task_definition.ml_app_task.arn
  desired_count   = 1 # Start with one instance of our application
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = data.aws_subnets.default.ids
    security_groups = [aws_security_group.ecs_tasks_sg.id]
    assign_public_ip = true # Assign public IP to find the container initially
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ml_app_tg.arn
    container_name   = "${var.project_name}-container"
    container_port   = 8000
  }

  # This ensures that the service waits for the ALB to be ready
  depends_on = [aws_lb_listener.http]
} 